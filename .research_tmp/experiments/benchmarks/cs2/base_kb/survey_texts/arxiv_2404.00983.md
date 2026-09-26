Continual Learning for Smart City: A Survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2404.00983v1 [cs.LG] 01 Apr 2024 
 
 

# Continual Learning for Smart City: A Survey

 
 
 Li Yang
 
    
 Zhipeng Luo*
 
    
 Shiming Zhang
 
    
 Fei Teng
 
    
 Tianrui Li
 † † thanks: 
Li Yang, Zhipeng Luo (the corresponding author), Shiming Zhang, Fei Teng, and Tianrui Li are with a) School of Computing and Artificial Intelligence, Southwest Jiaotong University, Chengdu 611756, China. b) Engineering Research Center of Sustainable Urban Intelligent Transportation, Ministry of Education. c) National Engineering Laboratory of Integrated Transportation Big Data Application Technology, Southwest Jiaotong University. d) Manufacturing Industry Chains Collaboration and Information Support Technology Key Laboratory of Sichuan Province, Southwest Jiaotong University, Chengdu 611756, Sichuan, P.R. China.
 † † thanks: E-mails: yangli_ef@my.swjtu.edu.cn, zpluo@swjtu.edu.cn, zhangshiming@my.swjtu.edu.cn, fteng@swjtu.edu.cn, trli@swjtu.edu.cn † † thanks: Manuscript received xxx; revised August xxx. 

 Abstract 
 
 With the digitization of modern cities, large data volumes and powerful computational resources facilitate the rapid update of intelligent models deployed in smart cities.
Continual learning (CL) is a novel machine learning paradigm that constantly updates models to adapt to changing environments, where the learning tasks, data, and distributions can vary over time.
Our survey provides a comprehensive review of continual learning methods that are widely used in smart city development.
The content consists of three parts:
1) Methodology-wise. We categorize a large number of basic CL methods and advanced CL frameworks in combination with other learning paradigms including graph learning, spatial-temporal learning, multi-modal learning, and federated learning.
2) Application-wise. We present numerous CL applications covering transportation, environment, public health, safety, networks, and associated datasets related to urban computing.
3) Challenges. We discuss current problems and challenges and envision several promising research directions. We believe this survey can help relevant researchers quickly familiarize themselves with the current state of continual learning research used in smart city development and direct them to future research trends.

 
 
 
 Index Terms:  Continual Learning, Smart City, Urban Computing, Deep learning.

 
 

## I Introduction 

 
 Nowadays, with advanced information technologies deployed citywide, large data volumes, powerful computational resources, and AI-based technologies are intelligentizing modern city development. Smart city research analyzes urban conditions and resident behaviors through urban data, aiming to enhance efficiency, sustainability, and quality of life for city residents.
Driven by fastly-iterated data collection and machine learning models, the development of a smart city also requires rapid and continuing progression.
To handle such a challenge, however, the standard practice is to purely retrain models with incrementally collected data, which can incur an affordable waste of time and computing resources.
Therefore, a more efficient solution called continual learning (CL) has recently drawn enormous attention from the machine learning community. In general, CL aims to help models rapidly learn new knowledge, which might come in the form of new data, new classes, or new domains.
The primary advantage of CL is that it only needs to update models, as if the new model has been retrained with both the previous and the new data. Such an update is usually low-key and can be done more frequently to achieve fast iterations.

 
 
 Fig. 1: An example of continual learning for graph learning. In (a) , a Model is firstly trained based on the data from Task A and then updated by Task B as Model *. Often, the new data distribution in Task B can be OOD. If the catastrophic forgetting problem is not handled well enough, the latest Model * can have degenerated performance on the previous Task A, shown in (b) . 
 
 
 The typical setting of continual learning is to learn a series of tasks sequentially.
In most CL scenarios, there is no universal assumption made about the data distribution among tasks. In fact, tasks can be generated from very different environments, so the new data can be out-of-distribution (OOD) or with the distribution shifted (DS) [ 1 , 2 ] .
The difference in data distributions poses a critical challenge in CL, that is, how the model can continually learn new knowledge while not forgetting what it has learned. It is conversely rephrased as the Catastrophic Forgetting (CF) problem [ 3 ] if models continually forget during learning. The key to dealing with this problem is to balance the plastic learning ability and the stable memory ability. The former refers to a model’s ability of continually acquiring new knowledge, while the latter refers to memorizing learned knowledge.
Fig. 1 illustrates a graph learning example where the graph, say, representing a city road network, can expand with time. A good model should be able to remember enough features or patterns of the previous graph structure (Task A) and then keep learning the new ones (Task B).

 
 
 Recently, we have experienced a surge of continual learning research in the machine learning community.
We performed an exhaustive search and count of CL-related papers published within the past five years at major conferences including CVPR, ACL, ICML, ICLR, NeurIPS, and journals like IEEE TPAMI, TKDE, and IJCV. The figure shows that the research activities on CL increased significantly from 29 publications in 2019 to more than 200 in 2023, resulting in a total increase of more than 10 times.
Early works began with iCaRL [ 4 ] , EWC [ 5 ] , and LwF [ 6 ] in 2017. Sooner, the computing vision community became the major field where CL prospered.
A lot of architecture-based, replay-based, and regularization-based methods were proposed and achieved remarkable results in different CL settings like incremental data, classes, domains, online learning, etc.
Such rapid growth of CL research provides an initial basis for studying more complex and practical problems in urban computing [ 7 ] .
During the same period, CL was gradually studied in various applications in smart cities, including transportation, environment, public health, public safety, public networks, auto-vehicles, and robots. We counted publications related to both CL and smart cities and plotted the statistics in Fig. 2 . This implies that smart city related CL works are in a strong uptrend and many promising challenges remain open.

 
 
 Fig. 2: The trend of continual learning research and its application to smart city research. Statistics are from Google Scholar over the past five years. 
 
 
 Related Surveys. The number of existing surveys on different aspects of continual learning isn’t small. For example, the most recent work by
Wang et al. [ 8 ] completed a very comprehensive review of the latest CL work as of 2023. Other reviews mainly focused on computer vision and natural language processing [ 1 , 9 , 10 , 11 , 12 ] . Specifically, Belouadah et al. [ 9 ] defined six expected properties of incremental learning algorithms and proposed a universal evaluation framework; De Lange et al. [ 10 ] proposed a widely recognized classification method for continual learning; Ke and Liu [ 11 ] provided a classification of CL methods in Natural Language Processing; Parisi et al. [ 1 ] presented the main challenges of continual learning from the perspective of biological principles; Zhou et al. [ 12 ] provided detailed experiments on the differences and advantages of various CL methods in the vision field. There have also been many reviews on graph neural networks that have received widespread attention [ 13 , 14 ] . In other fields, Shaheen et al. [ 15 ] provided a review of applications in real-world automation systems; Lesort et al. [ 16 ] investigated the CL methods and applications to robotics; and finally, Zhang and Kim [ 17 ] analyzed the CL methods and applications to recommendation systems.
Despite the rich number of surveys, none has particularly focused on the continual learning work done for urban computing or smart cities, where many important studies should not be overlooked.
Therefore, our survey is devoted to filling this gap by reviewing the recent edge-cutting CL work done for smart cities.

 
 
 Contributions. The contributions of our survey are summarized as follows:

 
 • 
 
 To the best of our knowledge, this is the first survey that reviews the latest continual learning work for smart cities. We summarize the advances from both application and methodology perspectives based on very rich literature.

 

 • 
 
 We categorize the major applications and their CL problem formulations in smart city scenarios. Meanwhile, we also list out numerous public datasets associated.

 

 • 
 
 We introduce various advanced continual learning frameworks common to smart cities that integrate other learning paradigms such as graph learning, temporal Learning, spatial-temporal learning, multi-modality learning, and federated learning.

 

 • 
 
 Lastly, we discuss existing challenges of continual learning research for smart cities and envision several promising future directions

 

 
 
 
 Organization. The rest of this survey is organized as follows. In Section II , we introduce the background of continual learning, including the basic task setting, common scenarios of CL in smart cities, and methods for overcoming catastrophic forgetting. In Section III , we summarize various applications of CL to smart cities and elaborate on how CL is used in different scenarios. In Section IV , we present advanced CL frameworks in combination with other learning paradigms. In Section V , we analyze the current open problems and challenges and suggest several few directions.
Finally, we conclude our survey in Section VI .

 
 
 Fig. 3 : A taxonomy of common continual learning methods to overcome catastrophic forgetting. (a). Regularization-based methods use additional regularization terms to prevent the important model parameters from deviating too much during sequential training. (b). Replay-based methods maintain a memory buffer or a generative model to replay the past data when training with new data. (c). Architecture-based methods memorize past knowledge by using previous network architecture and learn new knowledge by expanding the network’s architecture. The expanded network can be neuron-level or module-level. 
 
 
 

## II Background 

 
 We start by providing a brief introduction to smart cities and continual learning. We will cover the basic settings, scenarios, and a method taxonomy of continual learning. Also, we present two ways of categorization of CL methods applied to smart cities - Table I is based on different CL scenarios while Table II is based on different smart application fields of smart cities.

 
 

### II-A Smart Cities 

 
 A smart city is a technologically modern urban area that relies on information and communication technology (ICT) as its technical backbone and uses sensors and user devices deployed throughout the city to capture real-time data on the urban condition. This infrastructure enables enhanced urban management and crisis mitigation through coordinated planning and behavioral guidance. The overarching objective of smart cities is to make cities and human settlements inclusive, safe, resilient, and sustainable 1 1 
 1 
 
 
 
 https://www.un.org/sustainabledevelopment/cities/ . With the ongoing advancement of the Internet of Things, big data, and artificial intelligence, smart cities are progressing toward greater digitization, intelligence, and sustainable development. Hence, we are witnessing the rapid development of smart city applications across various fields, including transportation, environmental protection, public safety, healthcare, network management, urban robotics, and many others.

 
 
 

### II-B Task Settings and Scenarios of Continual Learning 

 
 Continual learning, sometimes also known as lifelong learning, has been defined differently in the literature, and there is no consensus on a very accurate definition. Nevertheless, we can define some of the essential elements necessary for our discussion. See below.

 
 
 DEFINITION 1 . 
 
 Base Model . The ultimate goal is to learn a supervised machine learning model f : 𝒳 → 𝒴 f:\mathcal{X}\rightarrow\mathcal{Y} that maps an input domain 𝒳 \mathcal{X} to an output domain 𝒴 \mathcal{Y} . Often, f f is a parametric model, and we parameterize it by θ \theta as f θ f_{\theta} . In most CL studies, f θ f_{\theta} is a neural network as it has both strong plastic and stable abilities. f θ f_{\theta} is continually trained through a sequence of tasks. 

 
 
 
 DEFINITION 2 . 
 
 Sequential Tasks . Suppose there is a series of tasks coming sequentially for a model to learn with. Each task has a task identifier τ ∈ { 1 , … , 𝒯 } \tau\in\{1,\dots,\mathcal{T}\} and a corresponding (labeled) dataset 𝒟 τ = { X τ , Y τ } \mathcal{D}_{\tau}=\{X_{\tau},Y_{\tau}\} . The dataset further consists of a training set 𝒟 τ t ​ r ​ a ​ i ​ n \mathcal{D}_{\tau}^{train} and a testing set 𝒟 τ t ​ e ​ s ​ t \mathcal{D}_{\tau}^{test} . 𝒯 \mathcal{T} can be either finite or infinite. The data 𝒟 τ \mathcal{D}_{\tau} are assumed to be sampled from an unknown distribution P τ ​ ( 𝒳 , 𝒴 ) P_{\tau}(\mathcal{X},\mathcal{Y}) , and we can informally say that P τ ​ ( 𝒳 , 𝒴 ) P_{\tau}(\mathcal{X},\mathcal{Y}) represents the knowledge of task τ \tau . 

 
 
 
 DEFINITION 3 . 
 
 Continual Learning . Continual Learning is a process that trains a base model f θ f_{\theta} through sequential tasks. The typical setting is that when task τ \tau comes, the model has been trained with the previous τ − 1 \tau-1 tasks, denoted by f θ ( τ − 1 ) f_{\theta}^{(\tau-1)} ; moreover, the model only has access to the data of the current task τ \tau , with which it is updated as f θ ( τ ) f_{\theta}^{(\tau)} . The goal of CL is to have the final model f θ ( 𝒯 ) f_{\theta}^{(\mathcal{T})} generalize well on all the tasks’ underlying distributions. To practically measure the generalizability, we test the model on all tasks’ test data, and a good model should achieve a low total loss, as shown in Equation ( 1 ). 

 

 
 | 
 θ ∗ = argmin 𝜃 ​ 1 𝒯 ​ ∑ τ = 1 𝒯 ℒ ⁡ ( f θ ( 𝒯 ) ​ ( X τ t ​ e ​ s ​ t ) , Y τ t ​ e ​ s ​ t ) \theta^{*}=\underset{\theta}{\mathrm{argmin}}\frac{1}{\mathcal{T}}\sum_{\tau=1}^{\mathcal{T}}\mathcal{L}(f_{\theta}^{(\mathcal{T})}(X_{\tau}^{test}),Y_{\tau}^{test}) | 
 | 
 (1) | 
 

 
 
 
 TABLE I: Continual learning scenarios and related work in smart cities. 
 
 
 Scenario | 
 Data Distribution | 
 Task Identifiability | 
 Publications | 

 
 TIL [ 2 ] | 
 
 
 
 𝒳 i = 𝒳 j , 𝒴 i = 𝒴 j \mathcal{X}_{i}=\mathcal{X}_{j},\mathcal{Y}_{i}=\mathcal{Y}_{j} 
 
 P i ​ ( 𝒳 , 𝒴 ) ≠ P j ​ ( 𝒳 , 𝒴 ) P_{i}(\mathcal{X},\mathcal{Y})\neq P_{j}(\mathcal{X},\mathcal{Y}) 
 | 
 τ \tau is available | 
 
 
 
 R2C-SVR-IL [ 18 ] ,
TF-Net [ 19 ] ,
IL-TFNet [ 20 ] ,
FpC [ 21 ] ,
TrafficStream [ 22 ] ,
STKEC [ 23 ] , 
 
 PAVEMENT [ 24 ] ,
IKASL [ 25 ] ,
ISVM [ 26 ] 
CEDUP [ 27 ] ,
SARDINE [ 28 ] ,
LIL [ 29 ] , 
 
 MCLNP [ 30 ] ,
SEIRD-IL [ 31 ] ,
FIRF [ 32 ] ,
HAR-IL [ 33 ] ,
Solver-CM [ 34 ] ,
TTD [ 35 ] , 
 
 ISVM [ 26 ] ,
ILAs [ 36 ] ,
KPGNN [ 37 ] ,
LL-ED [ 38 ] ,
FinEvent [ 39 ] ,
GraphSAIL [ 40 ] , 
 
 GDumb [ 41 ] ,
IGC [ 42 ] ,
STS-Rec [ 43 ] ,
SACDLD [ 44 ] ,
FIRE [ 45 ] ,
FMHP [ 46 ] , 
 
 GCMTP [ 47 ] ,
D-GSM [ 48 ] ,
SCL-PP [ 49 ] ,
CLTP-MAN [ 50 ] ,
CL-SGR [ 51 ] ,
AirLoop [ 52 ] , 
 
 DVS-IL [ 53 ] ,
ISTMM [ 54 ] ,
CAA-HRI [ 55 ] ,
TCBLS [ 56 ] 
 | 

 
 DIL [ 2 ] | 
 
 
 
 𝒳 i ≠ 𝒳 j , 𝒴 i = 𝒴 j \mathcal{X}_{i}\neq\mathcal{X}_{j},\mathcal{Y}_{i}=\mathcal{Y}_{j} 
 
 P i ​ ( 𝒳 i , 𝒴 i ) ≠ P j ​ ( 𝒳 j , 𝒴 j ) P_{i}(\mathcal{X}_{i},\mathcal{Y}_{i})\neq P_{j}(\mathcal{X}_{j},\mathcal{Y}_{j}) 
 | 
 τ \tau is optional | 
 
 
 
 FLCB [ 57 ] ,
FedSTIL [ 58 ] ,
DILRS [ 59 ] ,
DENet [ 60 ] ,
DR-EMR [ 61 ] 
 | 

 
 CIL [ 2 ] | 
 
 
 
 𝒴 i ∩ 𝒴 j = ∅ \mathcal{Y}_{i}\cap\mathcal{Y}_{j}=\emptyset 
 
 P i ​ ( 𝒳 i , 𝒴 i ) ≠ P j ​ ( 𝒳 j , 𝒴 j ) P_{i}(\mathcal{X}_{i},\mathcal{Y}_{i})\neq P_{j}(\mathcal{X}_{j},\mathcal{Y}_{j}) 
 | 
 τ \tau is unavailable | 
 
 
 
 IEL [ 62 ] ,
CITS [ 63 ] ,
Solver-CM [ 34 ] ,
Dynamic-IL [ 64 ] , 
 
 iCarl+ [ 65 ] ,
KCN [ 66 ] ,
EMP [ 67 ] 
 | 

 
 OCL [ 68 ] | 
 
 
 
 Based on TIL, DIL, or CIL 
 | 
 τ \tau is optional | 
 
 
 
 STMP [ 69 ] ,
ArcVideo [ 70 ] ,
GWR [ 71 ] ,
WISDOM [ 72 ] ,
ANNAIL [ 73 ] , 
 
 DIM-BLS [ 74 ] ,
CSTWPP [ 75 ] ,
HyperHawkes [ 76 ] ,
oHIML [ 77 ] ,
DEGC [ 78 ] ,
iEA [ 79 ] 
 | 

 
 
 The above definition of continual learning is quite general, and various specific scenarios can extend from it [ 2 , 8 ] . Below we introduce some common scenarios, which are distinguished according to their task identifiability and data distribution. Specifically, task identifiability means whether the task id τ \tau is available during the training or testing phase; and data distributions P τ ​ ( 𝒳 , 𝒴 ) P_{\tau}(\mathcal{X},\mathcal{Y}) , or even the domains 𝒳 \mathcal{X} and 𝒴 \mathcal{Y} , may differ from task to task. We summarize these scenarios and their publications related to smart cities in Table I .

 
 
 SCENARIO 1 . 
 
 Task-incremental learning (TIL) . In the TIL scenario, the base model can distinguish tasks by accessing their task identifiers in both the training and testing phases. The main challenge of TIL is to handle potentially different data distributions P τ ​ ( 𝒳 , 𝒴 ) P_{\tau}(\mathcal{X},\mathcal{Y}) among tasks. And the domains 𝒳 \mathcal{X} and 𝒴 \mathcal{Y} usually remain unchanged. 

 
 
 
 TIL is the most fundamental scenario and is widely studied in the area of smart cities, shown in Table I . Typical examples are TrafficStreem  [ 22 ] and STKEC  [ 23 ] that study traffic flow prediction problems in continually expanded cities. The year naturally becomes the task ID and the traffic flow in each new year has a large distribution shift. Accordingly, a base prediction model is updated yearly to adapt to the changes in traffic flow.

 
 
 SCENARIO 2 . 
 
 Domain-incremental learning (DIL) . In the DIL scenario, the key problem is to deal with potentially different input domains 𝒳 \mathcal{X} , while the output domain 𝒴 \mathcal{Y} usually remains unchanged. The task identifiers are optional. 

 
 
 
 DIL is often used to address environmental changes in smart cities. A typical application is remote sensing [ 59 , 60 ] , where a model aims to identify city construction or natural disasters. However, with the changing of air condition, sensor lifetime, complex background, etc., the domain shifts between tasks can lead to severe catastrophic forgetting. In DIL, the model needs to learn on various domains with sequential tasks.

 
 
 SCENARIO 3 . 
 
 Class-incremental learning (CIL) . In the CIL scenario, the CL model should predict new classes in forthcoming tasks, and in some extreme settings, the number or the name of new classes is even not informed in the testing phases. Hence, the output domains 𝒴 \mathcal{Y} can change. Meanwhile, the base model is unable to distinguish which task the new classes belong to through task identifiers. 

 
 
 
 The CIL scenario is a more recent research hotspot in the vision community, but it has not been fully investigated in smart city research. To name one of the few works, IEL [ 62 ] proposed an incremental urban garbage classification setting. With new tasks, the base model continues to identify new garbage classes to achieve good performance on every task.

 
 
 SCENARIO 4 . 
 
 Online Continual Learning (OCL) . In the OCL scenario, the data of each task come in a smaller granularity, in streaming instances or batches. OCL is built on the above three scenarios in an online fashion. 

 
 
 
 Online learning is a common setting in smart cities, such as traffic management system  [ 69 ] and recommendation system  [ 78 ] . Note that in some research, the term streaming data is used to denote online data, while in other cases, the term means incremental data.

 
 
 

### II-C Continual Learning Methods 

 
 There are many categories of continual learning methods, and here we introduce three common types. They are regularization-based, replay-based, and architecture-based methods, shown in Fig. 3 .

 
 

#### II-C 1 Regularization-based methods

 
 Regularization-based methods are characterized by adding regularization terms that explicitly control the model’s plastic and stable abilities. Usually, models trained by such methods have relatively stable memory but have limited capacity for learning new knowledge.
A typical implementation is to add a secondary penalty to the loss function, punishing the model’s important parameters for large deviations during its continual learning process.
To name a few, EWC  [ 5 ] was the first proposed work to reduce catastrophic forgetting (CF) by constraining important parameters. The importance was measured by the Fisher information matrix. SI  [ 80 ] calculated the importance of parameters online during the model training phase. The sensitivity of the cumulative loss function to the change of each parameter during the training process is used as the estimated importance. MAS  [ 81 ] used unlabeled samples to estimate the sensitivity of a neural network to parameter changes as an estimate of parameter importance.

 
 
 

#### II-C 2 Replay-based methods

 
 Another more effective way to overcome catastrophic forgetting is to memorize a small portion of previous data when training with new data. Replay-based methods typically maintain a memory buffer or a generative model to “replay” the past experience.
For example, Rebuffi et al.  [ 4 ] proposed for the first time the incremental classification and representation learning method iCaRL based on data replay. After learning each task, this method saves a small number of samples for each category for subsequent training. As the model can still see the saved representative data, the CF issue can be partly alleviated. Because of such an advantage, data replay techniques have been frequently used in numerous CL methods.

 
 
 

#### II-C 3 Architecture-based methods

 
 Neural networks have very flexible architectures, so people have proposed a new CL approach that can achieve zero forgetting, which is to dynamically expand the network. Usually, the network’s parameters are divided into parts that fit different tasks, so different knowledge is memorized without interfering. When necessary, the network can expand to allow learning new tasks.
Yan et al.  [ 82 ] proposed a dynamically expandable representation learning (DER) method with a modular deep classifier network containing a super feature extractor network and a linear classifier. Specifically, the super feature extractor network comprises multiple feature extractors of different sizes, adjusted for each incremental step. When faced with new classes, DER extends the network with new feature extractors while freezing the previous ones. Finally, features from all extractors are concatenated for class prediction.

 
 
 
 

### II-D Evaluation Metrics 

 
 The performance of continual learning methods should be evaluated from global-local and forgetting-memory perspectives. We introduce below some common evaluation metrics frequently used in the CL literature.

 
 
 Average Accuracy (AA) [ 83 ] measures the classification accuracy of the base model on both the past and the present tasks. The average accuracy of the current task τ \tau is defined as follows:

 

 
 | 
 A τ = 1 τ ​ ∑ k = 1 τ a τ , k A_{\tau}=\frac{1}{\tau}\sum_{k=1}^{\tau}a_{\tau,k} | 
 | 
 (2) | 
 

 where a τ , k a_{\tau,k} denotes the classification accuracy on the testing set 𝒟 k t ​ e ​ s ​ t \mathcal{D}^{test}_{k} of task k k after training on task τ \tau . AA suggests the global performance of the model on all the trained datasets { 𝒟 1 , 𝒟 2 , ⋯ , 𝒟 τ } \{\mathcal{D}_{1},\mathcal{D}_{2},\cdots,\mathcal{D}_{\tau}\} , but little does it reflects the local tasks and forgetting.

 
 
 Forgetting Measure (FM) [ 83 ] measures a model’s forgetting on all previous tasks after training on the current task. The average forgetting of the current task τ \tau is defined as follows:

 

 
 | 
 F τ = 1 τ − 1 ​ ∑ k = 1 τ − 1 f k τ F_{\tau}=\frac{1}{\tau-1}\sum_{k=1}^{\tau-1}f_{k}^{\tau} | 
 | 
 (3) | 
 

 where f k τ f_{k}^{\tau} denotes the difference between the maximum accuracy learned from all previous tasks and the accuracy after being updated by the current task. The forgetting at task k k after the CL model has been learned from the current task τ \tau is defined as:

 

 
 | 
 f k τ = max i ∈ { 1 , ⋯ , τ − 1 } ⁡ a i , k − a τ , k , ∀ k τ f_{k}^{\tau}=\max_{i\in\{1,\cdots,\tau-1\}}a_{i,k}-a_{\tau,k},\quad\forall\ k \tau | 
 | 
 (4) | 
 

 Apparently, large FM implies more forgetting.

 
 
 Backward Transfer (BWT) [ 84 ] measures the influence on how much the current learning task τ \tau affects the historical task k k ( k τ k \tau ). BWT calculates the mean of the accuracy differences of task k k between before and after the CL model trained on the 𝒟 τ t ​ r ​ a ​ i ​ n \mathcal{D}_{\tau}^{train} :

 

 
 | 
 BWT τ = 1 τ − 1 ​ ∑ k = 1 τ − 1 a τ , k − a k , k \mathrm{BWT_{\tau}}=\frac{1}{\tau-1}\sum_{k=1}^{\tau-1}a_{\tau,k}-a_{k,k} | 
 | 
 (5) | 
 

 A positive BWT indicates that the performance on task t t increases after the CL model has trained on task τ \tau . On the contrary, a large negative BWT means that forgetting happens on task k k after learning task τ \tau .

 
 
 Forward Transfer (FWT) [ 84 ] measures the influence of the current learning task τ \tau affecting the future task k k ( k τ k \tau ). FWT calculates the mean of the accuracy differences of task k k between the zero-shot learning and the randomly initialized network on the testing set 𝒟 k t ​ e ​ s ​ t \mathcal{D}_{k}^{test} . Here, the accuracy performance from the randomly initialized network is denoted as b k b_{k} :

 

 
 | 
 FWT τ = 1 τ − 1 ​ ∑ k = 2 τ a k − 1 , k − b k \mathrm{FWT_{\tau}}=\frac{1}{\tau-1}\sum_{k=2}^{\tau}a_{k-1,k}-b_{k} | 
 | 
 (6) | 
 

 A positive FWT shows that the CL model can transfer the knowledge from the preceding task to the current task, that is, the task k − 1 k-1 can improve the performance on task k k .

 
 
 In addition to the above metrics, there are many other ones to evaluate CL methods from different perspectives [ 8 , 11 ] . We will provide a detailed explanation of the metrics closely related to smart cities in the following sections.

 
 
 
 

## III Applications 

 
 In this section, we discuss task-specific challenges regarding the application of continual learning to smart cities. The literature we surveyed covers a wide range of areas. As shown in Tab. II , we include transportation, environment, public health, public safety, public networks, auto-vehicles, and robots in smart cities. The statistics indicate that recently there are great interests of applying CL methodologies to various areas of smart cities. Also, at the end of this section, we list numerous public datasets used in smart city research, as in Table III .

 
 
 TABLE II: Application domain categories of CL methods in smart city and their methods categories 
 
 
 Domain | 
 Subdomain | 
 Methods | 
 Regularization-based | 
 Replay-based | 
 Architecture-based | 

 
 Transportation | 
 Traffic Flow Prediction | 
 R2C-SVR-IL [ 18 ] | 
 ✓ | 
 | 
 | 

 
 STMP [ 69 ] | 
 ✓ | 
 | 
 | 

 
 TF-Net [ 19 ] | 
 ✓ | 
 | 
 | 

 
 IL-TFNet [ 20 ] | 
 ✓ | 
 | 
 | 

 
 FpC [ 21 ] | 
 ✓ | 
 | 
 | 

 
 TrafficStream [ 22 ] | 
 ✓ | 
 ✓ | 
 | 

 
 STKEC [ 23 ] | 
 ✓ | 
 ✓ | 
 | 

 
 iETA [ 79 ] | 
 ✓ | 
 ✓ | 
 | 

 
 Traffic Video Analysis | 
 ArcVideo [ 70 ] | 
 | 
 ✓ | 
 | 

 
 PAVEMENT [ 24 ] | 
 ✓ | 
 | 
 | 

 
 FLCB [ 57 ] | 
 ✓ | 
 | 
 | 

 
 FedSTIL [ 58 ] | 
 | 
 ✓ | 
 | 

 
 Traffic Trajectory Analysis | 
 IKASL [ 25 ] | 
 | 
 | 
 ✓ | 

 
 GWR [ 71 ] | 
 | 
 | 
 ✓ | 

 
 GCMTP [ 47 ] | 
 | 
 ✓ | 
 | 

 
 D-GSM [ 48 ] | 
 | 
 ✓ | 
 | 

 
 CLTP-MAN [ 50 ] | 
 | 
 ✓ | 
 | 

 
 CL-SGR [ 51 ] | 
 | 
 ✓ | 
 | 

 
 SCL-PP [ 49 ] | 
 ✓ | 
 ✓ | 
 | 

 
 Environment | 
 Air Quality Prediction | 
 CEDUP [ 27 ] | 
 | 
 ✓ | 
 | 

 
 WISDOM [ 72 ] | 
 ✓ | 
 | 
 | 

 
 Pollution Classification | 
 IEL [ 62 ] | 
 ✓ | 
 | 
 | 

 
 Remote Sensing | 
 SARDINE [ 28 ] | 
 | 
 | 
 ✓ | 

 
 LIL [ 29 ] | 
 | 
 | 
 ✓ | 

 
 DILRS [ 59 ] | 
 ✓ | 
 | 
 ✓ | 

 
 Public Health | 
 / | 
 MCLNP [ 30 ] | 
 ✓ | 
 | 
 | 

 
 ANNAIL [ 73 ] | 
 ✓ | 
 | 
 | 

 
 SEIRD-IL [ 31 ] | 
 | 
 ✓ | 
 | 

 
 Public Safety | 
 Human Activity Recognition | 
 FIRF [ 32 ] | 
 ✓ | 
 | 
 | 

 
 CITS [ 63 ] | 
 | 
 ✓ | 
 | 

 
 HAR-IL [ 33 ] | 
 | 
 ✓ | 
 | 

 
 Solver-CM [ 34 ] | 
 | 
 ✓ | 
 | 

 
 DIM-BLS [ 74 ] | 
 | 
 ✓ | 
 ✓ | 

 
 TTD [ 35 ] | 
 ✓ | 
 | 
 | 

 
 Natural Disasters Detection | 
 DENet [ 60 ] | 
 ✓ | 
 | 
 | 

 
 Power System Analysis | 
 Dynamic-IL [ 64 ] | 
 | 
 ✓ | 
 | 

 
 CSTWPP [ 75 ] | 
 ✓ | 
 | 
 | 

 
 Public Networks | 
 Internet Traffic Classification | 
 ILAs [ 36 ] | 
 ✓ | 
 | 
 | 

 
 ISVM [ 26 ] | 
 | 
 ✓ | 
 | 

 
 iCarl+ [ 65 ] | 
 ✓ | 
 ✓ | 
 | 

 
 Social Event Representation | 
 HyperHawkes [ 76 ] | 
 ✓ | 
 | 
 | 

 
 oHIML [ 77 ] | 
 ✓ | 
 | 
 | 

 
 DR-EMR [ 61 ] | 
 | 
 | 
 ✓ | 

 
 Social Event Detection | 
 KCN [ 66 ] | 
 ✓ | 
 | 
 | 

 
 KPGNN [ 37 ] | 
 | 
 | 
 ✓ | 

 
 FinEvent [ 39 ] | 
 | 
 | 
 ✓ | 

 
 LL-ED [ 38 ] | 
 ✓ | 
 ✓ | 
 | 

 
 EMP [ 67 ] | 
 ✓ | 
 ✓ | 
 | 

 
 Recommendation | 
 GraphSAIL [ 40 ] | 
 ✓ | 
 | 
 | 

 
 SACDLD [ 44 ] | 
 ✓ | 
 | 
 | 

 
 GDumb [ 41 ] | 
 | 
 ✓ | 
 | 

 
 FIRE [ 45 ] | 
 | 
 ✓ | 
 | 

 
 FMHP [ 46 ] | 
 | 
 ✓ | 
 | 

 
 STS-Rec [ 43 ] | 
 | 
 | 
 ✓ | 

 
 DEGC [ 78 ] | 
 | 
 | 
 ✓ | 

 
 IGC [ 42 ] | 
 ✓ | 
 | 
 | 

 
 Robots | 
 / | 
 AirLoop [ 52 ] | 
 ✓ | 
 | 
 | 

 
 ISTMM [ 54 ] | 
 ✓ | 
 | 
 | 

 
 CAA-HRI [ 55 ] | 
 ✓ | 
 | 
 | 

 
 DVS-IL [ 53 ] | 
 ✓ | 
 ✓ | 
 | 

 
 TCBLS [ 56 ] | 
 | 
 | 
 ✓ | 

 
 

### III-A Transportation 

 
 The transportation systems of modern cities can be extremely complex. Physically, there are now very rich commuting options for residents to choose from, such as bus, subway, taxi, biking, or walking; technologically, many digital devices such as GPS, sensors, and cameras are deployed everywhere. Hence, perceiving, recording, and managing rich and changing transportation information becomes a non-trivial challenge. So people are leveraging continual learning to deal with (near) real-time traffic changes and incidents. Below we mainly introduce three sub-areas: traffic flow prediction, traffic video analysis, and trajectory analysis.

 
 

#### III-A 1 Traffic flow prediction

 
 Prediction of traffic flow on urban roads is one important task in intelligent transportation systems (ITSs). Traffic flow can be denoted as 1) the speed and counts of vehicles crossing some sections or 2) traffic inflow and outflow in certain regions, as shown in Fig. 4 . Usually, both forms above are modeled as graph structures 𝒢 \mathcal{G} and share the same prediction goals defined as follows.

 

 
 | 
 { X t − M + 1 , ⋯ , X t ; 𝒢 } ​ → f θ ​ { X t + 1 , ⋯ , X t + N } \{X_{t-M+1},\cdots,X_{t};\mathcal{G}\}\overset{f_{\theta}}{\rightarrow}\{X_{t+1},\cdots,X_{t+N}\} | 
 | 
 (7) | 
 

 where the X t X_{t} denotes the features of the traffic flow at a timestamp t t , and f θ f_{\theta} denotes the prediction model.

 
 
 
 
 
 (a) Station traffic flow 
 
 
 (b) Region traffic flow 
 
 Fig. 4: Two different types of data recording for traffic flow 
 
 
 There is a long history and many well-studied models on urban traffic flow prediction, from traditional time series methods, such as Auto-Regressive Integrated Moving Average (ARIMA) [ 85 ] , Support Vector Regression (SVR) [ 86 ] and Gradient Boosting Decision Tree (GBDT) [ 87 ] , to deep learning methods [ 88 , 89 , 90 ] . Such models are mainly devoted to solving static data and thus become very limited in dealing with dynamic transportation networks and live flows.

 
 
 Introducing continual learning into traffic flow prediction can effectively mitigate the above problems.
In early works, CL was used to solve the concept drift problem in non-stationary data  [ 18 , 69 ] .
Later, some other works focused on meeting the real-time requirements of traffic flow prediction and reducing training costs through CL  [ 19 , 20 , 21 ] .
More recent works have combined graph neural networks (GNNs) and CL to perform traffic flow prediction. These works centered more on solving the catastrophic forgetting problem. We mention some representative works below.

 
 
 TrafficStream  [ 22 ] first utilized CL to solve the problems of traffic network expansion and traffic flow evolution in a long-term streaming network. The network is defined as a sequence of evolving snapshots: 𝒢 = ( 𝒢 1 , 𝒢 2 , ⋯ , 𝒢 𝒯 ) \mathcal{G}=(\mathcal{G}_{1},\mathcal{G}_{2},\cdots,\mathcal{G}_{\mathcal{T}}) . The goal of TrafficStream is to learn a series of functions Ψ = ( Ψ 1 , Ψ 2 , ⋯ , Ψ 𝒯 ) \Psi=(\Psi_{1},\Psi_{2},\cdots,\Psi_{\mathcal{T}}) to predict traffic flow series:

 

 
 | 
 Ψ i ∗ = arg ⁡ min Ψ i ⁡ ‖ Ψ 𝒯 ​ ( X 𝒯 − ) − X 𝒯 + ‖ 2 \Psi_{i}^{*}=\arg\min_{\Psi_{i}}\left\|\Psi_{\mathcal{T}}(X_{\mathcal{T}-})-X_{\mathcal{T}+}\right\|^{2} | 
 | 
 (8) | 
 

 where the X 𝒯 − X_{\mathcal{T}-} and X 𝒯 + X_{\mathcal{T}+} denote the past and the future (ground-truth) traffic flow data, respectively.

 
 
 STKEC  [ 23 ] also dealt with constantly evolving and expanding traffic networks. Specifically, incremental nodes and edges representing new sensor stations are continuously added to the topological graph over time. STKEC follows similar problem settings and notion definitions as TrafficStream.
iETA [ 79 ] is another work studying evolving traffic conditions, yet not on expanding traffic networks. They were all motivated by the fact that periodically retraining the prediction model is expensive while updating the predictors based on incremental data is more realistic.

 
 
 

#### III-A 2 Traffic video analysis

 
 Traffic video analysis can be used for traffic monitoring, traffic analysis, traffic scheduling, and other applications  [ 91 , 92 ] .
We categorize a few tasks related to traffic video analysis, including video classification  [ 93 ] , object detection  [ 94 ] , and semantic segmentation  [ 95 ] .
Usually, the above works primarily focus on static tasks so have not addressed dynamically evolving tasks. For instance, the model needs to recognize vehicles and pedestrians under different weather conditions, as well as newly introduced trucks and trains on the road. Below we introduce some CL-related works.

 
 
 VPaaS  [ 96 ] classified Video Analysis Systems into two categories, client-driven and cloud-driven . VPaaS posed three challenges regarding cloud-driven methods, bandwidth and latency in cloud transmission, data drift in fixed and pre-trained models, and inherent pipeline problems.
In contrast, Ekya  [ 97 ] focused on client-driven methods, also known as edge computing. To overcome the data drift problem on edge devices, Ekya utilizes CL techniques to maintain the performance of compressed models across many tasks.
Besides the data drift problem, ArcVideo [ 70 ] further considered the human labeling cost and edge storage cost incurred by using CL methods.
Specifically, new video frames and manual labels are required constantly, which increases human labor. Meanwhile, the replay-based approach consumes a lot of memory space, which is unfriendly for the storage in edge devices. C-EC [ 98 ] leveraged the advantages of both client-driven and cloud-driven methods, helping edge models learn from real-time video and prevent forgetting caused by data drift.
Based on visual analysis tasks, PAVEMENT [ 24 ] proposed to compensate for the environmental impact of existing vehicle detection methods by integrating non-video sensors such as vibration sensors and Doppler sensors. Meanwhile, a CL technique is used to reduce the cost of human labeling and calculation overhead in the training process.

 
 
 In addition to vehicle detection, pedestrian detection is also an important application of traffic video analysis. Crowd-counting can be used to predict the number of people in images or videos and help manage crowded scenes in real-time. FLCB [ 57 ] proposed a lifelong crowd-counting task to obtain an optimal crowd-counting function on a series of 𝒯 \mathcal{T} domain datasets. These different domains usually come from different locations of cameras, such as on a street, in a park, or in a gym. FLCB attempted to achieve a globally optimal performance on all domains through the CL technology, rather than a specific domain.

 
 
 Person re-identification (ReID) is another application in pedestrian detection. Person ReID aims to identify the same person from cameras with different perspectives. Researchers first noticed that classic methods have many limitations in the real-world environment. For example, ReID data are constantly acquired from new locations or domains.
GwFReID [ 99 ] believed there are three major challenges in the lifelong ReID task - that are zero-shot problems, incremental domains and classes, and imbalance classes. Meanwhile, AKA [ 100 ] proposed that the Lifelong ReID needs to have the generalization ability to the unseen classes, and the classification ability to the fine-grained inter-class appearance variations. Subsequently, PTKP [ 101 ] took the fast domain adaptation ability as the goal of lifelong person ReID. PTKP pointed out that GwFReID and AKA do not consider the issues of task-wise domain gap, so their models can not learn task-shared knowledge very well. Following the above existing works, MEGE [ 102 ] further studied the generalized representation of lifelong ReID models without forgetting the knowledge, and considered underlying adjacent relations between samples. Meanwhile, KRC [ 103 ] regarded the lifelong ReID as a fine-grained open-set problem. KRC believed that existing lifelong person ReID works are mainly focused on preventing the forgetting problem. However, it is also important that the model can have positive FWT and BWT by the CL technology.
Based on the above person ReID works, FedSTIL [ 58 ] further considered privacy and security issues during data transmission and model training, thus a federated learning technique was used to make the edge clients obtain lifelong learning ability. In the setting of Federated Lifelong Person ReID, each edge client c c learns from streaming datasets 𝒟 τ c \mathcal{D}_{\tau}^{c} , which denotes the τ \tau -th incremental datasets at edge client c c . The goal of FedSTIL was to achieve lifelong learning for a single client and joint learning across clients while avoiding the sharing of sensitive information among edge clients and the center cloud.

 
 
 

#### III-A 3 Trajectory analysis

 
 While lots of traffic analysis concentrates on traffic flow or congestion, there are also research works on comprehensive traffic condition analysis, a domain we refer to as Trajectory Monitoring .
Trajectory data can be collected from GPS, cell towers, and Wi-Fi to help city managers, drivers, or pedestrians know better about the traffic conditions from both citywide and individual perspectives.
By surveying the previous works, we categorize the applications as trajectory prediction [ 104 , 105 ] , trajectory clustering [ 106 , 107 ] , and Path Planning.

 
 
 IKASL [ 25 ] proposed an incremental trajectory cluster algorithm to represent hyper-dimensional trajectories and to capture the time variation of traffic trajectories. IKASL segments and profiles raw road traffic data into distinct trajectory clusters, continually refining these profiles over time to adapt to evolving traffic patterns. This approach offers a clear understanding of traffic flow dynamics and serves as a foundation for effective traffic management strategies.
GWR  [ 71 ] studied the mobility of human populations based on the trajectory datasets. In this work, the authors focused on resolving the forgetting problem caused by the concept shift of streaming data and further implemented a detection algorithm of the concept shift and assessment of their harmful effects.

 
 
 In addition to analyzing traffic trajectories at a citywide scale, there has been more attention on analyzing trajectories from an individual perspective, such as agents like vehicles and pedestrians. Trajectory Prediction is a prominent area of interest within this research domain. The trajectory prediction task involves forecasting the future paths of one or more agents using historical trajectory data and current road conditions.
Accurate trajectory prediction is crucial for enhancing the safety of autonomous driving systems, enabling autonomous vehicles (AVs) to anticipate and avoid pedestrians and other vehicles effectively [ 108 , 109 ] . Fig. 5 shows a trajectory prediction task under complex road conditions.

 
 
 Fig. 5: A crossroads scenario for trajectory prediction. The solid line represents the agent’s historical trajectory, and the dashed line represents the predicted trajectory. We observe a green vehicle and pedestrian stationary, while an orange vehicle and pedestrian are in motion. Their trajectories exhibit a degree of predictability. Conversely, the trajectory of the yellow vehicle and pedestrian has just started, introducing significant uncertainty. The objective of trajectory prediction is to anticipate future behavior within such intricate scenarios precisely. 
 
 
 SILA [ 110 ] first raised the problem that the offline setting and batch learning setting limit the ability of pedestrian trajectory prediction models in terms of responding to environment changes and their generalization ability. Therefore, through the CL technology, the models are allowed to update the flexibly while adding the incremental few-shot data. Meanwhile, the features not exposed to historical data also enhance privacy. In SILA, an algorithm was designed to learn and update motion primitives and transitions incrementally.
SCL [ 49 ] aimed to help autonomous mobile robots realize the accurate prediction of the trajectories from pedestrians around the robots. Moreover, SCL proposed a self-supervised continual learning framework to achieve an online trajectory prediction.
CLTP-MAN [ 50 ] and CL-SGR [ 51 ] were the following studies that further explore different CL techniques to solve the CF problem in pedestrian trajectory predictions.
GCMTP [ 47 ] was the earliest study on multi-agent trajectory prediction in the CL setting. Compared to single-agent trajectory prediction, multi-agent trajectory prediction coordinates multiple agents, usually referring to vehicles, simultaneously. So that their predicted trajectories do not collide with each other. GCMTP believed that the forgetting problem in the trajectory prediction task would take the form of the location changing. That is, multi-agent behaviors from a new interaction location may be very different from old interactions, so the model may prefer to learn the trajectory pattern from the current location and forget the historical locations.
IPCC-TP [ 111 ] proposed to utilize the Incremental Pearson Correlation Coefficient to obtain an optimal multi-agent trajectory prediction.
GRTP [ 112 ] designed a Lifelong Vehicle Trajectory Prediction Framework for reliable autonomous driving, hoping the model can maintain consistent performance under different traffic circumstances.
Similarly, D-GSM [ 48 ] was proposed to predict vehicle trajectory for autonomous driving in the CL scenarios.

 
 
 
 

### III-B Environment 

 
 Environment-friendly is critical to modern urban development, and technologies should constantly improve upon this target. To achieve long-term environmental monitoring and protection, continual learning, because of its cost-effectiveness in nature, is gradually being used in various environmental applications in smart cities. Below we introduce air quality control, pollution control, and remote sensing.

 
 

#### III-B 1 Air quality prediction

 
 With urbanization and industrialization, air quality has become one critical environmental problem in cities. To monitor and protect air quality, many deep learning methods [ 113 , 114 ] have been proposed to achieve the air quality prediction.
To model municipal-level carbon emissions, CEDUP [ 27 ] explored carbon emission distribution through incremental learning modeling, which can prevent provincial-level patterns from being forgotten while learning about municipal-level patterns.
Climate prediction plays a crucial role in the environmental monitoring of smart cities. Based on the combination of CL and spatio-temporal prediction models, SMART [ 72 ] designed a Weight Incremental Spatio-Temporal Multi-Task Learning Algorithm (WISDOM), an incremental learning algorithm to achieve incremental learning on temporal-spatial data and can easily adapt to the dynamic spatio-temporal domain. Furthermore, SMART attempted to address the limitation of traditional methods where models cannot incorporate known spatio-temporal knowledge. For example, when models are learning local climate patterns, they should also be able to use larger-scale weather patterns.

 
 
 

#### III-B 2 Pollution classification

 
 Urban pollutants accumulate with the operation and production of cities. Pollution, such as urban garbage, air pollution, and water pollution, if not cleaned up in time, can hurt people’s health and happiness. There have been a lot of DNN-based methods to predict and classify pollution [ 115 , 116 ] .
Due to the increase of urban garbage, recycling of garbage resources has become a challenging problem. In a practical application environment, garbage groups will change over time, and it is difficult to ensure the accuracy of classification by using static models. This problem can be partially solved by CL. Han et al. [ 62 ] propose the IEL algorithm, which can improve the garbage classification accuracy and generalization ability of incremental learning.

 
 
 

#### III-B 3 Remote sensing

 
 Remote sensing has played an important role in urban planning, natural disaster monitoring, and safety prevention. Remote sensing images captured by artificial satellites can provide continuous weather observation, building observation, and ecological environment observation covering a large area of the city, which is very useful for detailed monitoring and environmental quality assessment.
However, with the accumulation of new data and the drift of data distribution, the traditional static model is difficult to deal with in a complex environment, so using CL is an important method to solve the above challenges.

 
 
 In recent years, some work has introduced CL into remote sensing.
SARDINE [ 28 ] was an earlier work in this field, in which the proposed model’s incremental learning capability provides additional advantages when addressing variations in spatial contexts across widely dispersed regions, as observed in satellite imagery.
LIL [ 29 ] proposed two new challenge issues of remote Sensing in CL, which are the task-sharing feature extractor large-capacity issue and task-specific module redundancy issue.
DILRS [ 59 ] instead focused on the domain-incremental problem. Specifically, the domain shift in the remote sensing task includes inconsistent class distributions across various regions, along with disparities in spatial resolutions and spectral divergence of specific object categories across different sensors.
Unlike the classification task, the counting task is a regression problem. DMD [ 117 ] discusses the application of CL for counting systems of urban objects such as buildings, vehicles, and ships.

 
 
 
 

### III-C Public Health 

 
 Citizens’ health in city management comes in the first place. Take epidemics as an example, when a global pandemic such as COVID-19 threatens people’s health, predicting uncertainty in the pandemic is a huge challenge. COVID-19 is characterized by rapid variation and infection, and traditional pre-training models are insufficient to handle the fast dynamic change. Hence, continual learning can be used here for its ability to adapt to the uncertainty caused by environmental changes, especially in non-stationary environments.
Farooq et al. [ 73 ] proposed a DNN-based and data stream-guided real-time online incremental learning algorithm (ANNAIL) to study the transmission dynamics and prevention mechanism for COVID-19. Their work can help forecast the pandemic and suggest relevant policies.
For a similar motivation, SEIRD-IL [ 31 ] proposed an incremental learning approach for online predictions of epidemic diseases based on a dynamic ensemble method. And MCLNPs [ 30 ] studied the uncertainty problem of a CL method in the non-stationary environment of the epidemic disease prediction task.

 
 
 

### III-D Public Safety 

 
 The operation of modern cities is supported by multiple infrastructures such as food, power, water, gas, and network supplies. Monitoring and detection of faults in these systems are necessary to guarantee public safety. In addition, cities may suffer from sudden disasters like accidents and natural disasters, and thus we also need to maintain real-time intelligent models that have learned the latest changes.

 
 

#### III-D 1 Human activity recognition

 
 One important aspect of public safety is human activity recognition(HAR). HAR recognizes human activities through wireless sensor data and can help prevent extreme public incidents from happening. In the real world, the sensor or edge equipment faces challenges such as privacy protection, new behavior or new user recognition, and power efficiency.
Recently, many works are applying the continual learning logic to address the above challenges. Some researchers applied CL to reduce the training time of edge equipment [ 32 , 33 ] , update the model without the historical privacy data [ 34 ] , identify new users and their new actions [ 63 , 74 , 34 ] , or solve the sensor temporal data increment and catastrophic forgetting problems [ 35 ] .

 
 
 

#### III-D 2 Natural disasters detection

 
 Natural disasters such as wildfires, earthquakes, and floods are devastating to people’s lives and properties. Detecting these natural disasters in time can help dramatically reduce potential losses. Some researchers proposed to detect disasters by using social media or satellite imagery [ 118 , 119 ] .
To develop fire detection models based on sensor data, it is necessary to balance the real-time performance and computational efficiency of the model, and CL is a highly effective way to tackle such as problem. Wang et al. [ 60 ] proposed a domain-incremental fire detection method to enable incremental updates of models by continuously learning heterogeneous data.

 
 
 

#### III-D 3 Power system analysis

 
 City supplying systems’ stability has become a more important issue since the complexity of these systems increases, which in part leads to a higher rate of failures. Therefore, it is necessary to identify and classify the system faults. In a real system, a complete fault dataset cannot be obtained at once, which requires the model to have the ability to identify new faults. Veerakumar et al. [ 64 ] introduced CL to power systems with the ability to address catastrophic forgetting.
CSTWPP [ 75 ] trained the model continually, enabling the model to serve online and solve the problems of processing non-stationary new data and consuming excessive computing resources for repeated training.

 
 
 
 

### III-E Public Networks 

 
 Many activities of modern cities take place in cyberspace. Therefore, both the local and global networks need to be securely monitored and managed. Here we discuss network traffic analysis, social events analysis, and recommendation system design from the view of continual learning.

 
 

#### III-E 1 Internet traffic analysis

 
 The ability to identify flows and their related protocols in network traffic classification is necessary for many applications, such as security and quality of service (QoS). To enable traffic classification models to obtain large-scale data and real-time processing capabilities, CL is one of the important tools. Sun et al. [ 26 ] introduced an incremental SVMs (ISVM) model to reduce the high training cost of memory and CPU, such that the traffic classifier can achieve quick updates. Bovenzi et al. [ 65 ] utilized iCarl to also solve the high-cost problem of retraining models. Eldhai et al. [ 36 ] proposed four incremental learning algorithms to effectively identify concept drifts while using less memory and time.

 
 
 

#### III-E 2 Social event analysis

 
 In the tasks of social event analysis, a realistic and challenging problem is to continually learn time-to-event models in an ever-changing environment while retaining previously learned knowledge. Dubey et al. [ 76 ] and Vijayaraghavan [ 61 ] proposed continual learning-based representation learning approaches to address the above challenges. Zhao et al. [ 77 ] proposed an incremental multi-source feature learning algorithm to quickly learn the new missing patterns in real-time without retraining the whole model.
Social event detection also plays an important role in smart city management. There are already lots of works on how to continuously learn models for new event classes while not forgetting previously learned knowledge under the constraints of computational costs and storage budgets [ 66 , 37 , 38 ] . And more works followed in this line - EMP [ 67 ] introduced episodic memory prompts to explicitly retain the learned task-specific knowledge to relieve CF; FinEvent [ 39 ] proposed using an incremental learning framework to help the model continuously acquire, preserve, and extend the semantic space, and make use of its advantages to deal with the actual problems.

 
 
 

#### III-E 3 Recommendation systems

 
 Recommendation systems have thrived in many parts of public networks. Retraining the model in practical applications is very time-consuming, and directly fine-tuning the model can also lead to catastrophic forgetting. To solve such problems, many studies have focused on recommendation systems. Also, graph neural networks are introduced as many user-item, user-user, and item-item relationships can be well represented by graphs. Therefore, recommendation systems based on continual graph learning [ 40 , 41 , 44 , 42 , 45 , 43 , 46 , 78 ] have become an important research direction.

 
 
 
 

### III-F Robots 

 
 Although intelligent robots aren’t widely used in smart cities, they indirectly contribute to city management, production, and our daily lives. CL can play a crucial role in robot perception, recognition, action, and other processes.
Simultaneous localization and mapping (SLAM) systems are one of the most important components of modern robots. AirLoop [ 52 ] utilized CL to minimize forgetting when training loop closure detection models incrementally. For recognition, Lungu et al. [ 53 ] proposed a hand symbol recognition system that can incrementally train on the platforms with limited resources and recognize new symbols without forgetting the old symbols.
Besides, Human-robot interaction is another big topic. Kanazawa et al. [ 54 ] introduced a collaborative robot that can update the model adaptively. Social robots need a model with the ability to deal with individual differences and emotional changes in the process of interaction with people and to learn new emotions during the update of models [ 120 , 121 , 55 , 122 ] .
In addition, there are a large number of specialized equipment in smart cities. Accurately predicting the remaining useful life of such equipment can increase the production efficiency. TCBLS [ 56 ] introduced CL in an online learning scenario where new data are constantly acquired. Faced with situations where newly acquired data and prediction accuracy are inadequate, online machine learning of new data and nodes can be implemented to adaptively update and upgrade the network.

 
 
 TABLE III: Open datasets for continual learning in smart cities. 
 
 
 Domain | 
 Datasets | 
 Links | 

 
 Transportation | 
 PEMS3-Stream | 
 https://github.com/AprLie/TrafficStream | 

 
 VicRoads | 
 https://vicroadsopendata-vicroadsmaps.opendata.arcgis.com | 

 
 PeMS | 
 http://pems.dot.ca.gov | 

 
 Madrid | 
 https://datos.madrid.es/portal/site/egob | 

 
 Barcelona | 
 https://opendata-ajuntament.barcelona.cat/data/en/dataset/itineraris | 

 
 Gdansk | 
 https://doi.org/10.34808/8xkq-7714 | 

 
 Turin | 
 https://doi.org/10.5194/isprs-annals-IV-4-W7-3-2018 | 

 
 IARAI | 
 https://proceedings.mlr.press/v133/kopp21a.html | 

 
 UTD19 | 
 https://utd19.ethz.ch/ | 

 
 DashCam | 
 https://arxiv.org/pdf/2102.03012.pdf | 

 
 Porto | 
 https://ieeexplore.ieee.org/abstract/document/6532415 | 

 
 ShanghaiTech | 
 https://doi.org/10.1109/CVPR.2016.70 | 

 
 UCF-QNRF | 
 https://doi.org/10.1007/978-3-030-01216-8_33 | 

 
 NWPU-Crowd | 
 https://doi.org/10.1109/TPAMI.2020.3013269 | 

 
 JHU-Crowd++ | 
 https://doi.org/10.1109/ICCV.2019.00131 | 

 
 Market-1501 | 
 https://doi.org/10.1109/ICCV.2015.133 | 

 
 PKU-ReID | 
 https://arxiv.org/pdf/1605.02464.pdf | 

 
 PersonX | 
 https://arxiv.org/abs/1812.02162 | 

 
 Prid2011 | 
 http://dx.doi.org/10.1007/978-3-642-21227-7_9 | 

 
 DukeMTMC-reID | 
 https://doi.org/10.48550/arXiv.1609.01775 | 

 
 Environment | 
 FASDD | 
 https://doi.org/10.57760/sciencedb.j00104.00103 | 

 
 Huawei Cloud | 
 https://ieeexplore.ieee.org/abstract/document/9435085 | 

 
 SMART-data | 
 https://github.com/Jianpeng-Xu/TKDE-SMART | 

 
 Public Health | 
 COVID | 
 https://www.kaggle.com/fireballbyedimyrnmom/us-counties-covid-19dataset | 

 
 Public Safety | 
 WISDM | 
 https://archive.ics.uci.edu/ml/machine-learning-databases/00507 | 

 
 Anguita dataset | 
 https://sensor.informatik.uni-mannheim.de/#dataset_realworld | 

 
 HAPT | 
 https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf | 

 
 Public Networks | 
 MIRAGE-2019 | 
 http://traffic.comics.unina.it/mirage/app list.html | 

 
 TOR dataset | 
 https://www.scitepress.org/PublishedPapers/2017/61056/61056.pdf | 

 
 Lifelong Social Events Dataset | 
 https://pralav.github.io/lifelong eventrep?c=10 | 

 
 Yelp | 
 https://www.kaggle.com/datasets/yelp-dataset/ | 

 
 Meme | 
 https://snap.stanford.edu/data/memetracker9.html | 

 
 FewEvent | 
 https://arxiv.org/pdf/1910.11621 | 

 
 MAVEN | 
 https://arxiv.org/pdf/2004.13590 | 

 
 ACE05-EN | 
 https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/lrec2004-ace-program.pdf | 

 
 Taobao2014 | 
 https://tianchi.aliyun.com/dataset/dataDetail?dataId=46 | 

 
 Nefix | 
 https://academictorrents.com/details/9b13183dc4d60676b773c9e2cd6de5e5542cee9a | 

 
 Auto-vehicle | 
 ETH | 
 http://vision.cse.psu.edu/courses/Tracking/vlpr12/PellegriniNeverWalkAlone.pdf | 

 
 PUCY | 
 https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2007.01089.x | 

 
 inD | 
 https://arxiv.org/pdf/1911.07602 | 

 
 INTERACTION | 
 https://arxiv.org/pdf/1910.03088.pdf; | 

 
 SDD | 
 https://infoscience.epfl.ch/record/230262/files/ECCV16social.pdf | 

 
 Robots | 
 TartanAir | 
 https://arxiv.org/pdf/2003.14338 | 

 
 Nordland | 
 https://arxiv.org/pdf/1808.06516 | 

 
 RobotCar | 
 https://doi.org/10.1177/0278364916679498 | 

 
 AffectNet | 
 https://arxiv.org/pdf/1708.03985 | 

 
 
 

### III-G Open Datasets 

 
 Lastly, we summarize a list of open datasets related to continual learning and smart city work in Table III . Some datasets are specially designed for CL tasks, which means the datasets are manually divided into several parts, as if they are sequential tasks. For example, PEMS3-Stream is built from PeMS by using seven years’ data with each year being a task. In other research, authors group multiple datasets as one CL dataset. In this case, authors usually consider one as a training dataset and others as new sequential datasets for the latter can have significantly different distributions from the first one.

 
 
 
 

## IV Advanced Continual Learning Frameworks 

 
 In this section, we introduce some of the latest continual learning frameworks combined with other machine learning paradigms. Such combinations are as expected since continual learning itself is sort of an “add-on” process that is built on top of other learning tasks, which can vary a lot. For our purpose, we will review those advanced CL learning frameworks closely related to smart city research. They are continual graph learning, temporal continual learning, spatial-temporal continual learning, multi-modality continual learning, and federated continual learning.

 
 

### IV-A Continual Graph Learning 

 
 Graphs are widely used for network data representation such as citation networks [ 123 , 124 ] , social networks [ 125 , 126 ] , traffic networks [ 127 , 128 , 129 ] , and knowledge graphs [ 130 , 131 , 132 ] . In a graph, nodes represent entities and edges represent their relationships [ 133 ] . Analysis of graphs focuses on tasks such as node classification, link prediction, graph classification, and so on. Graph Neural Networks (GNNs) are a type of graph analysis method based on deep learning. With the steady progress of research, GNNs’ focus is gradually changing from static graphs to dynamic ones. However, current research on dynamic graphs mainly focuses on changes on node features or edges, with less consideration on the addition of nodes and sub-graphs, which may cause catastrophic forgetting.

 
 
 Fig. 6: Continual graph learning deals with the increment or decrement nodes and edges in a continuously evolving graph. The original graph as Task 1 is marked in blue. After two evolutions, the new graphs with incremental nodes and edges are marked in orange and green. 
 
 
 Continual graph learning (CGL) studies how to implement CL methods on graph-structured data. While many CL methods are particularly designed for vision tasks, they cannot be directly applied to graph data, which are in a unique non-Euclidean data structure. The specificity of GCL is explained in Fig 6 . In general, GCL faces three challenges:

 
 • 
 
 How to detect and represent new instances on a graph, including node-level and graph-level information, and keep accumulating new knowledge.

 

 • 
 
 How to achieve efficient graph learning without using the entire graph.

 

 • 
 
 How to avoid catastrophic forgetting and memorize what has been learned from previous graphs.

 

 
 
 
 Currently, the majority of CGL works focus on the first challenge. For example, DyGNN [ 134 ] proposed a general framework that can keep updating node information by capturing the sequential information of edges. At the same time, Han et al. [ 135 ] focused on the same problem that how to deal with new or unseen data by using GNNs. Daruna et al. [ 136 ] attempted to represent new nodes and edges on the Knowledge Graph. Later, other works such as [ 22 ] and [ 23 ] focused on modeling graph expansions, specifically referring to the expansion of an urban road network. FILDNE [ 137 ] studied how to apply static graph representation methods to dynamically incremental graphs. Recently, the unbalance between new and old classes on graphs is another research area, e.g. [ 23 ] .

 
 
 Streaming graph data is a common scenario in real-world graph learning applications, for example, to add or delete a node/edge in social networks with changing relationships. Wang et al. [ 138 ] were the early ones dealing with streaming graph data and overcoming forgetting by using data replaying and model regularization. In their following work, [ 139 ] proposed a streaming GNN by using the generative replay method. Other research covers more areas. Wang et al. [ 140 ] used knowledge graphs to model streaming events. DiCGRL [ 141 ] proposed a disentangle-based continual graph representation learning to alleviate the forgetting problem. IncreSTGL [ 142 ] proposed an Incremental Spatio-Temporal Graph Learning framework that has an incremental graph representation learning module to refine and update query-POI interaction graphs in an online incremental fashion. FGN [ 143 ] converted a CGL problem to a regular graph learning problem so that GNN can inherit the lifelong learning techniques developed for convolutional neural networks.

 
 
 Besides, some works focus on tackling catastrophic forgetting. To keep the long-term preference of users, GraphSAIL [ 40 ] implemented a graph structure preservation strategy that explicitly preserves each node’s local structure, global structure, and self-information.
ER-GNN [ 144 ] stored knowledge from previous tasks as experiences and replayed them when learning new tasks to mitigate the catastrophic forgetting issue. Three experience node selection strategies, called mean of feature, coverage maximization, and influence maximization, were proposed to guide the process of selecting experience nodes.
Another work using the replay-based method was [ 145 ] . In this work, an incremental training method for lifelong learning on graphs was presented and a new measure based on k-neighborhood time differences to address variances in the historical data is proposed. Further, a Hierarchical Prototype Networks (HPNs) [ 146 ] was presented, which extracts different levels of abstract knowledge in the form of prototypes to represent the continuously expanded graphs.

 
 
 In addition, some works focus on the challenge regarding economic costs, like computational and memory consumption, which can be expensiveve especially in training deep models. IGCN [ 147 ] was proposed to learn new data and avoid the computational cost incurred when retraining GCN models.
Ahrabian [ 41 ] presented a novel framework for incrementally training GNN models with an experience reply technique. After that, a novel Graph Structure Aware Contrastive Knowledge Distillation for Incremental Learning in recommence systems [ 44 ] was proposed to focus on the rich relational information in the recommendation context, and the contrastive distillation formulation was combined with intermediate layer distillation to inject layer-level supervision. More recently, a causal incremental graph convolution (IGC) [ 42 ] approach was proposed, which consists of two new operators named IGC and colliding effect distillation (CED) to estimate the output of full graph convolution.

 
 
 Finally, other works consider CGL under more complex conditions, such as few-shot or multi-modality. The abovementioned tasks all assume sufficient labels, but in reality, there are many scenarios with insufficient labels of new nodes and edges, i.e., graph few-shot class-incremental learning (GFSCIL) problem. Geometer [ 148 ] learned and adjusted the attention-based prototypes based on the geometric relationships of proximity, uniformity, and separability of representations. A teacher-student knowledge distillation and biased sampling strategy were proposed to further mitigate the catastrophic forgetting and unbalanced labeling in GFSCIL. Under the same problem setting, a Hierarchical-Attention-based Graph Meta-learning framework (HAG-Meta) [ 149 ] presented a task-sensitive regularizer calculated from task-level attention and node class prototypes to mitigate overfitting onto either novel or base classes. To combine multiple modalities such as visual and textual features, the Multi-modal Structure-evolving Continual Graph Learning (MSCGL) model [ 150 ] was proposed to simultaneously take social information and multi-modal information into account to build the multi-modal graphs.
In addition, some important benchmark works were presented. Cart et al. [ 151 ] gave a benchmark for graph classification by experimenting in a robust and controlled framework. CGLB [ 152 ] systematically studied the task configurations in different application scenarios and developed a comprehensive continual graph learning benchmark curated from different public datasets.

 
 
 TABLE IV: Advanced continual learning frameworks used in smart city research. 
 
 
 Learning Frameworks | 
 Model | 
 Specific Task | 

 
 Continual Graph Learning | 
 DyGNN [ 134 ] | 
 Computer Vision | 

 
 FILDNE [ 137 ] | 
 Representation Learning | 

 
 DiCGRL [ 141 ] | 
 Representation Learning | 

 
 IncreSTGL [ 142 ] | 
 Query-POI Matching | 

 
 FGN [ 143 ] | 
 Representation Learning | 

 
 GraphSAIL [ 40 ] | 
 Recommender Systems | 

 
 ER-GNN [ 144 ] | 
 Node Classification | 

 
 HPNs [ 146 ] | 
 Node Classification | 

 
 IGCN [ 147 ] | 
 Recommender Systems | 

 
 IGC [ 42 ] | 
 Recommender Systems | 

 
 Geometer [ 148 ] | 
 Node Classification | 

 
 HAG-Meta [ 149 ] | 
 Node Classification | 

 
 MSCGL [ 150 ] | 
 Multi-modal Node Classification | 

 
 CGLB [ 152 ] | 
 Benchmark | 

 
 Temporal Continual Learning | 
 CLA [ 153 ] | 
 Finance Decision-making | 

 
 Solver-CM [ 34 ] | 
 Multivariate Time Series | 

 
 HyperHawkes [ 76 ] | 
 Time-to-Event Modeling | 

 
 TTD [ 35 ] | 
 Temporal data classification | 

 
 Spatial-temporal Continual Learning | 
 IL-GMM [ 54 ] | 
 Planning Motion | 

 
 R2C [ 18 ] | 
 Traffic Volume Prediction | 

 
 IKASL [ 25 ] | 
 Trajectory Clustering | 

 
 CGM [ 47 ] | 
 Trajectory Prediction | 

 
 SARDINE [ 28 ] | 
 Remote Sensing | 

 
 CSTWPP [ 75 ] | 
 Wind Power Forecasting | 

 
 WISDOM [ 72 ] | 
 Multi-task Learning | 

 
 TF-Net [ 19 ] | 
 Traffic Flow Prediction | 

 
 IL-TFNet [ 20 ] | 
 Traffic Flow Prediction | 

 
 TrafficStream [ 22 ] | 
 Traffic Flow Prediction | 

 
 STKEC [ 23 ] | 
 Traffic Flow Prediction | 

 
 Multi-modality Continual Learning | 
 CLiMB [ 154 ] | 
 Benchmark | 

 
 MSCGL [ 150 ] | 
 Node Classification | 

 
 CCMR [ 155 ] | 
 Image-text Retrieval | 

 
 BMU-MoCo [ 156 ] | 
 Video-Language Modeling | 

 
 VQACL [ 157 ] | 
 Visual Question Answering | 

 
 Mod-X [ 158 ] | 
 Vision-Language Representation | 

 
 LMC [ 159 ] | 
 Knowledge Graph Construction | 

 
 Federated Continual Learning | 
 FedWeIT [ 160 ] | 
 Image Classification | 

 
 CFeD [ 161 ] | 
 Image Classification | 

 
 FedSpace [ 162 ] | 
 Image Classification | 

 
 FedSTIL [ 58 ] | 
 Person Re-identification | 

 
 FpC [ 21 ] | 
 Urban Traffic Forecasting | 

 
 
 

### IV-B Temporal Continual Learning 

 
 Time series and temporal sequential data are commonly used data structures in dynamic systems. Such data are collected from sensors and continuously change with time, and this variation mainly manifests as data distribution shift.
There are several tasks for temporal sequential data,
such as time series forecasting [ 163 ] (including long-term time series forecasting and short-term time series forecasting),
time series imputation [ 164 ] ,
time series anomaly detection [ 165 ] ,
and time series classification [ 35 ] . Among them, forecasting tasks and imputation tasks are more likely to use generative or regression models, while anomaly detection and classification are more likely to use discriminative or classification models.

 
 
 Deep neural network-based methods have been broadly developed for temporal sequence data mining for a long time. In the past few years, methods that based on Recurrent Neural Network(RNN) [ 166 , 167 , 168 , 169 ] are the mainstream of temporal sequence data mining. Soon afterward, with the extreme success of Transformers [ 170 ] in natural language processing and computer vision, a lot of work has explored applying the attention mechanism in Transformer to temporal sequence data mining [ 163 , 171 , 172 ] . Recently, MLP-based methods [ 173 , 174 , 175 ] have also achieved good results in terms of accuracy and computational speed.

 
 
 However, in real-world applications, data are collected over time, requiring repeated learning of new tasks. When learning incremental tasks, standard deep learning-based methods suffer from distribution shifts and catastrophic forgetting. Hence, modern models should be able to learn incremental knowledge continuously, and introducing the CL philosophy into temporal data mining is a promising direction. We thus propose a category named Temporal Continual Learning (TCL) to classify these works.

 
 
 As a typical example of continuous time series forecasting, DoubleAdapt [ 176 ] design a two-fold adaptation, called data adaption and model adaption against distribution shift in stock price trend forecasting task. The data adaption transfers the feature distributions and posterior distribution to their corresponding agent distribution, which can close the gap between incremental data and test data. To optimize this two-fold adaptation, DoubleAdapt utilized ideal from MAML [ 177 ] , which uses a two-step updating method to find an optimum in all tasks.

 
 
 Continuous time series classification tasks have attracted more attention. The reason is that many well-studied CL strategies targeted classification tasks, and many class incremental learning methods from the computing version field can be migrated to continuous time series classification tasks.
For example, TTD [ 35 ] proposed a continual learning training method called Temporal Teacher Distillation. In the stage where the model learns the incremental task, TTD trains and tunes the model on each sequence of training datasets. To overcome catastrophic forgetting in LSTM, TDD proposed three hypotheses and combined them with distillation Loss and experience replay.
Gupta et al. [ 34 ] proposed a novel modularized neural network architecture to handle variable input dimensions, with two main modules: a core dynamics module comprising an RNN that models the underlying dynamics of the system, and a conditioning module using a GNN that adjusts the activations of the core dynamics module for each time-series based on the combination of sensors available, effectively exhibiting different behavior depending on the available sensors.
Dubey et al. [ 76 ] proposed HyperHawkes, a sequence descriptor-conditioned hypernetwork-based neural Hawkes process that can generate sequence-specific parameters to address continual learning of event sequences. The neural Hawkes process comprises two building blocks - RNN and feedforward neural network (FNN) and overcomes forgetting by incorporating a regularization on the hypernetwork parameters such that it penalizes any change to the FNHP parameters produced from old sequences.

 
 
 

### IV-C Spatial-temporal Continual Learning 

 
 Most data generated in cities come in spatial-temporal structures, which means that data not only have spatial features but also can change over time. The most critical challenge lies in their complex dependencies on various spatial and temporal indicators. Another practical issue is heterogeneity, meaning that the distribution drift can be not only in the time dimension but also in the space dimension. For these reasons, traditional methods that only focus on time or space, such as Auto-regressive Integrated Moving Average (ARIMA), Gradient Boosting Decision Tree (GBDT), ResNet, or AlexNet, can hardly tackle these problems completely. Fortunately, the recent advance in spatial-temporal data mining offers many spatial-temporal models, such as ConvLSTM, PredRNN, ST-ResNet, STGCN, and Graph WaveNet. Readers are referred to a few surveys [ 178 , 179 ] to overview such frontiers.

 
 
 However, even advanced models become limited when dealing with streaming data, incremental tasks, or new classes. It is because they mainly aim at handling one task at a certain time point, without considering any adaptation to future changes. For example, when a city expands, the model designed for the original urban structure may not adaptively capture the features of the new region; more importantly, the new region can produce a complex spatial-temporal dependence on the old region over time.

 
 
 To solve the problems above, recent works propose to apply continual learning techniques to spatial-temporal models. Kanazawa et al. [ 54 ] used a Gaussian mixture model (GMM) to capture human motion patterns. They extracted spatial features from people’s working states and temporal features from moving states. And finally, they implemented a new incremental learning algorithm for the GMM to adapt to new changes.
Xiao et al. [ 18 ] proposed a regression to classification (R2C) framework that can train any ensemble algorithms on linear and non-linear SVRs. There are three steps in the proposed framework. The first step is to construct classification datasets from regression datasets; then, by integrating them with the proposed Learn++ algorithms, multiple classifiers can be integrated; finally, a regression model is constructed upon the classifiers, which is updated incrementally.

 
 
 Trajectory data naturally contain spatial-temporal information. Bandaragoda et al. [ 25 ] proposed an approach using hyper-dimensional computing to transform variable-length trajectories of commuter trips into fixed-length high-dimensional vectors, with the benefit of being incrementally learned over time.
For trajectory prediction, Ma et al. [ 47 ] proposed a graph-neural-network-based continual multi-agent trajectory prediction framework. It consists of three modules, a graph-neural-network-based predictor, an episodic memory buffer, and a conditional-variational-autoencoder-based generative memory module.

 
 
 As for other applications, SARDINE [ 28 ] modeled the overall spatio-temporal prediction problem as the prediction of derived remote sensing imagery. To consider the influence of temporally evolving spatial features in the neighborhood of each pixel, SARDINE proposed a technique called layer growing for better modeling the spatial features’ evolution in various spatial contexts.
CSTWPP [ 75 ] was proposed to forecast wind power, and a CNN has been used for spatial-temporal feature extraction and wind power prediction. The whole training procedure of CSTWPP is divided into two steps: the first step is to train the CSTWPP model offline on the training set using SGD; and the second step is to train the CSTWPP online through incremental learning on the testing set, termed online training.
In WISDOM [ 72 ] , the spatial-temporal data are assumed to be periodically augmented with a new data chunk. The goal was to adapt the existing models without rebuilding them from scratch when new observations were available. To ensure that the model parameters and latent factors do not vary significantly from their previous values, a smoothness criterion was added as a constraint to the objective function.
TF-Net [ 19 ] used incremental learning rather than batch learning to perform model training. Their traffic flow dataset was divided into multiple sub-train sets, and the model was iteratively trained with each set. When the model stopped to improve for a few epochs, it would be transferred to the next sub-train set for training. Similarly, IL-TFNet [ 20 ] proposed an incremental learning-based CNN-LTSM model, with a similar training phase.

 
 
 TrafficStream [ 22 ] is a typical spatial-temporal CL method, which used a simple GNN as a surrogate model for complex traffic flow forecasting methods. In TrafficStream, new nodes with their 2-hops neighbors were used to construct a sub-graph for mining the influence of network expansion. Further, an algorithm based on JS Divergence detected existing nodes whose traffic patterns changed significantly. To consolidate the previous knowledge, historical nodes of traffic networks were replayed with weighted smoothing constraints imposed on the current training model.
To improve the TrafficStream, STKEC [ 23 ] developed a new framework containing two major components. One was an influence-based knowledge expansion strategy, used for integrating newly evolved traffic patterns of expanding road networks; and the other was a memory-augmented knowledge consolidation mechanism, which was to consolidate the old knowledge of the previous road network.

 
 
 

### IV-D Multi-modal Continual Learning 

 
 Various sensors deployed in cities can collect numerous types of urban information, recorded as vehicle trajectories, traffic flows, satellite data, user feedback, etc. These types of information are recorded as multi-modal heterogeneous data, including but not limited to images, videos, texts, maps, graphs, and point clouds. To this end, the trend of AI is transitioning from developing single-modal models to universal models that can process multiple modes of data. Thus multi-modal machine learning (MML) has emerged [ 180 , 181 , 182 ] .
In general, the objectives of MML include constructing a unified feature representation [ 183 ] , achieving semantic alignment between different modes [ 181 ] , integrating and fusing the feature representation of different modes [ 184 ] , generating new data of one mode based on another [ 185 ] , and transferring knowledge between modes [ 186 ] . The downstream tasks of MML usually include visual question answering [ 187 ] , natural language for visual reasoning [ 188 ] , vision-language retrieval [ 189 ] , and visual dialogue [ 190 ] .

 
 
 Very recently, large-scale pre-training techniques [ 191 , 192 ] are prevailing. For example, CLIP [ 193 ] used natural language as a training signal to the image data, on the basis of a large dataset (400 million [image, text] pairs) and big backbone models (ResNet-50 [ 194 ] or ViT [ 195 ] ). Moreover, a simple comparative learning method [ 196 ] was added to make CLIP a multi-modal model with zero-shot capabilities, thus achieving good generalization on new unseen data. After CLIP, many large-scale pre-training multi-modal studies have been proposed [ 197 , 198 , 199 ] .

 
 
 Fig. 7: A multi-modal continual learning (MCL) framework. In current literature, there are two types of tasks in MCL - one is modal incremental, and the other is the usual domain or class incremental. As shown in the upper part of the figure, the modal incremental problem requires the base model to continuously learn new modes of data. In the lower part, the other task aims to train the model with good generalization ability in multiple domains or in out-of-distribution situations. 
 
 
 Despite the success, the catastrophic forgetting problem in many multi-task and multi-modal learning works has not been well solved. Therefore, multi-modal continual learning (MCL) has been proposed. An overview framework of MCL is shown in Fig. 7 .
As examples, CLiMB [ 154 ] introduced continual learning in multi-modality benchmark to facilitate the study of CL in vision-and-language tasks with deployment to multi-modal and uni-modal tasks. A learning problem was formulated where a model was first trained on sequentially arriving vision-and-language tasks, referred to as upstream continual learning, and then transferred downstream to low-shot multi-modal and uni-modal tasks.
Cai et al. [ 150 ] presented a Multi-modal Structure-evolving Continual Graph Learning (MSCGL) model, which can adaptively explore model architectures without forgetting history information. MSCGL extracted multi-modal features using ViT and BERT in the data preprocessing stage, then designed different graph structures for different modalities, and used neural architecture search (NAS) to determine the new network architecture for searching the model with the best memory ability.

 
 
 

### IV-E Federated Continual Learning 

 
 Fig. 8: A stander federated continual learning setting. Each client updates its local model on a sequence of tasks. For instance, client A updates a local model in tasks 1, 3, 4, and 6 with incremental samples or classes. Once updated, client A will transmit the parameters to the center server. Meanwhile, a global model is continually trained whenever new parameters arrive, and the updated parameters of the global model will be transmitted to each local model, marked in red line. 
 
 
 In reality, data are often generated from different sources. For example, traffic data come from traffic-management departments, and air quality data are from environment-protection departments. In addition, the Internet of Things (IoT), a new distributed technology, has facilitated the collection of data from numerous edge devices. Therefore, in modern situations, unified models need to process different data chunks from various sites in a decentralized way.
To this end, federated learning (FL) paradigm has been proposed to solve the challenges above [ 200 , 201 , 202 ] . By utilizing FL techniques, models can process different sources of data in a distributed way when facing data isolation and privacy problems.

 
 
 Federated learning was first proposed by Macahan et al in 2017 [ 203 ] , where the classic FedAVG was presented to train distributed data simultaneously. More specifically, local data are first trained by local clients to construct initial models. Then, the models parameters’ gradients are sent to the central server, and the global model’s gradients are aggregated by weighted average. The FedAVG update is shown in Eq. 9 .

 

 
 | 
 ω t + 1 = ∑ k = 1 K m k m ​ ω t + 1 k \omega_{t+1}=\sum_{k=1}^{K}\frac{m_{k}}{m}\omega_{t+1}^{k} | 
 | 
 (9) | 
 

 where ω t + 1 \omega_{t+1} denotes the global parameters in the server, ω t + 1 k \omega_{t+1}^{k} is the local parameters in the client k k , and m k m \frac{m_{k}}{m} represents the proportion of client k k in the total sample of M M clients participating in training. Finally, the gradients of the updated global model are returned to the clients, and the above steps repeat until the global model converges.

 
 
 More recently, federated learning has been combined with continual learning, the so-called federated continual learning (FCL) [ 204 , 160 ] , to handle the challenges of dealing with incremental data or classes, non-stationary data distributions, and catastrophic forgetting problems that also happen in FL. A standard FCL setting is shown in Fig. 8 .
Formally, Suppose a sequence of tasks with a center server 𝒮 \mathcal{S} and a set of distributed clients 𝒞 \mathcal{C} . There are two goals of FCL - to minimize the average loss on all tasks and also to learn a global performer on all clients and the server. This can be written as Eq. 10 .

 

 
 | 
 min θ ∑ t = 1 T ∑ c ∈ 𝒞 N c t N t ℒ ( 𝒟 c t ; θ ) \min_{\theta}\sum_{t=1}^{T}\sum_{c\in\mathcal{C}}\frac{N_{c}^{t}}{N^{t}}\mathcal{L}(\mathcal{D}_{c}^{t};\theta) | 
 | 
 (10) | 
 

 Where θ \theta is model parameters, t t is the ID of each task, and c c is the ID of each client. N c t / N t N_{c}^{t}/N^{t} represents the portion of training samples from client c c to all clients in task t t . 𝒟 c t \mathcal{D}_{c}^{t} is the training dataset of task t t and client c c .

 
 
 There are two main challenges in FCL, inter-client interference and communication-efficiency . The first challenge says that historical knowledge may decrease the performance of other clients, and the second one assumes that unlimited streams of tasks can cause intractable performance overhead.
To mitigate the intra-task forgetting and inter-task forgetting problems in FCL, CFeD [ 161 ] used a surrogate dataset that is publicly available and employed a distillation loss for clients to review old tasks and for the server to perform model aggregation.
FedWeIT [ 160 ] focused on the communication challenge. It separated general knowledge and task-adaptive knowledge by decomposing the local parameters into dense-base parameters and sparse task-adaptive parameters.
FedSpace [ 162 ] proposed a new setting called Asynchronous Federated Continual Learning (AFCL), where each client has its own CL procedure till a communication step that updates all the clients to the server. To implement AFCL, FedSpace developed multiple procedures including Server Fractal Pre-training, Prototype Aggregation, Contrastive Representation Loss, and Server Aggregation.

 
 
 The above FCL studies ignored some particularities of the distributed devices in urban environments.
To alleviate catastrophic forgetting, FedSTIL [ 58 ] designed a lifelong learning framework for distributed edge clients. This edge framework included Extraction Layer, Adaptive Layer, and Prototype Rehearsal Storage. Specifically, the Extraction Layer was used for extracting raw data ( X i ( t ) , Y i ( t ) ) ∈ D c ( t ) (X_{i}^{(t)},Y_{i}^{(t)})\in D_{c}^{(t)} for edge client c c on the t t -th round to the extracted prototype set P c ( t ) P_{c}^{(t)} . To alleviate forgetting, the Prototype Rehearsal method was used to periodically sample and store some extracted prototypes of incremental tasks in local storage with the nearest-mean-of-exemplars strategy [ 4 ] . In the Adaptive Layer, the global knowledge from the server and local knowledge from clients are used for training by the following Eq. 11 

 

 
 | 
 θ c = B c ⊙ α c + A c \theta_{c}=B_{c}\odot\alpha_{c}+A_{c} | 
 | 
 (11) | 
 

 Where A c A_{c} is the knowledge learned from local incremental tasks, B c B_{c} is the base parameters with the spatial-temporal knowledge learned from global parameters, and α c \alpha_{c} is the attention parameters to capture the task-specific knowledge.

 
 
 Lastly, FpC [ 21 ] was proposed from the perspective of server-less FL based on the p2p paradigm [ 205 ] and was the first to apply FCL to urban traffic flow prediction. In FpC, the model parameters were updated by going through all the edge clients with a random path. And by using FCL, the edge devices in smart cities enjoy many advantages like better privacy protection, lower communication overhead and latency, and smaller memory usage and energy consumption.

 
 
 
 

## V Discussions 

 
 We have presented the basic and advanced methods, applications, and datasets of continual learning related to smart cities. Despite the remarkable advances, there remain quite many challenges to address. In this section, we discuss some of the important ones: continual large models, multi-model continual learning, continual learning in open world, privacy and security issues, and model explainability.

 
 

### V-A Continual Large Models 

 
 Large Language Models (LLMs) such as GPT [ 206 ] , PaLM [ 207 ] , and LLaMA [ 208 ] have demonstrated their strong learning intelligence and generalization.
The state-of-the-art LLMs have hundreds of billions of parameters, which makes them intractable to re-train or hard to update.

 
 
 From the perspective of continual learning, different from traditional small pre-trained models, several critical issues hinder the evolution of LLMs.
First, the primary limitation results from the high cost (computational, financial, time) of re-training or updating models. Therefore, more lightweight training methods and efficient CL strategies need to be devised.
The second issue is attributed to completely different training approaches for LLMs. Often, new data for fine-tuning large models are dramatically small compared to the model size, so this inhibits large models from effectively acquiring new knowledge.
The third challenge arises because LLMs are designed for strong generalization ability, likely on numerous domains. However, current CL methods primarily focus on maximizing knowledge within a single or few domains. Consequently, continual learning methods for LLMs must be particularly designed such that LLMs can learn extensive knowledge across various domains.
Furthermore, the absence of rigid analysis also poses a significant challenge to continual learning for LLMs. There is an urgent need to design insightful theories to guide the CL methods for LLMs.
Lastly, ethical issues matter as well. LLMs must adhere to social morality and legal requirements, which can be achieved through methods such as alignment learning. This is a critical issue during the continual learning phase of LLMs.

 
 
 In light of such problems, continual language learning (CLL) or continual language model (CLM) has started to emerge [ 209 , 210 , 211 , 212 ] , as well as a survey on continual learning for LLMs [ 213 ] .
As for smart city development, the main bottleneck lies in lacking well-developed large unified models. Many data in smart cities are generated in different forms like graphs, spatial-temporal, or distributed structures. However, the key technology such as large-scale pre-training techniques has not yet been adapted to such data structures. We believe that the trend would be to develop large unified models for processing different types of urban data. Afterward, how to continually update such large models would be an important future direction.

 
 
 

### V-B Multi-modality Continual Learning 

 
 In recent years, multi-modal learning is steadily developing and has flourished with large-scale pre-train models, such as CLIP [ 193 ] , Stable Diffusion, BLIP-2 [ 214 ] , and multi-modality LLMs [ 199 , 215 , 216 , 217 , 218 ] .
In our view, smart city multi-modal CL faces the following challenges. The first one is the lack of high-quality and large-scale urban multi-modal datasets. In smart cities, such data are in fact rich, but because of many practical reasons such as privacy and security issues, obtaining such data has always been a major issue. Therefore, there is a great need for building more smart cities’ multi-modal datasets and large-scale datasets.
The second challenge lies in the continual learning part. Many pre-trained multi-modal models can be directly applied to smart cities, such as text, image, and video processing. However, when it comes to some data modes unique to urban computing, such as traffic and weather data, existing multi-modal models become insufficient, and performing only fine-tuning can hardly meet the actual needs. Some existing studies are limited to specific tasks and modes [ 154 , 150 , 156 ] , so there is a large room for developing multi-modal models with continual learning abilities in smart city environments.

 
 
 

### V-C Continual Learning in Open World 

 
 In our view, continual learning in open worlds can be viewed from two perspectives: self-education and lifelong education.
Self-education assumes that continual learning models must have the capability to acquire new knowledge autonomously . That is, these models must independently detect new objects (OOD data or tasks) in the environment, understand, and convert them into memorized knowledge without human intervention or guidance. Ideally, such models are supposed to self-plan learning paths.
As for lifelong education, we expect the model to possess the capability to explore new environments and integrate the new knowledge acquired into existing knowledge bases. Essentially, the model needs to continuously expand and adapt to environmental changes.
There are already studies on the concept of continual learning in open worlds, such as Continual Reinforcement Learning [ 219 , 220 , 221 , 39 , 222 ] and open-world continual learning [ 223 , 224 , 225 , 226 ] . However, these studies primarily concentrate on addressing the OOD problem, leaving much room of realizing the aforementioned objectives of self-education and lifelong education. Additionally, they didn’t particularly focus on urban applications, where there are more intricate scenarios and demands.

 
 
 

### V-D Privacy and Security 

 
 As mentioned earlier, privacy and security issues have become one major obstacle hindering the development of unified models.
On privacy, continual learning needs to collect user data periodically, but the rights may not always be authorized; furthermore, privacy leakage may occur everywhere during data collection, transmission, or storage processes. On security, distributed devices in cities are vulnerable to attack. If individual data are not securely protected, then data aggregation would become infeasible.

 
 
 From one aspect, the security research community needs to define more comprehensive criteria for evaluating privacy and security attacks in real systems. Also, stronger protection techniques such as robust encryption algorithms should be developed, which are required for secure data collection and exchange. From another aspect, the continual learning community needs to research how intelligent models (central or distributed) can perform continual learning securely, especially when dealing with incomplete data pieces because of privacy or security constraints. We have seen works of federated continual learning that are devoted to this regard [ 21 , 58 , 160 , 161 , 227 , 228 , 162 ] . However, more complex and practical situations will motivate more secure continual learning scenarios and strategies.

 
 
 Moreover, a significant security concern arises from continual learning itself. Due to the nature of CL, a learning model may have accumulated a vast amount of historical knowledge, which could potentially be illicitly extracted through various methods, leading to the leakage of critical information. Hence, there is a need to explore continual learning methods that can actively and selectively forget. Active and controlled forgetting is not only beneficial for enhancing the model’s memory efficiency and performance [ 229 ] but also important for safeguarding privacy and enhancing security protection.

 
 
 

### V-E Explainability 

 
 Explainable artificial intelligence (XAI) is one general trend in the machine learning community. Advanced models such as deep neural nets have significantly boosted the performance but also become much poorer in explainability, and many of their generated results are not interpretable.
For city management, the behaviors of any decision models before deployment must be well understood and interpreted by not only their developers but also the city administrators [ 230 , 231 ] .
However, the majority of smart city CL works have focused on overcoming catastrophic forgetting and improving predictive power by using much more complex models (such as expandable deep networks).
A strong side effect would be that the explainability of such complex models becomes very unclear, and the study of explainability is no longer only a matter for basic models but also for their continual learning strategies. There is very little research on this regard. For example, ICICLE [ 232 ] proposed an interpretable class-incremental learning framework that can reduce the interpretability concept drift.
Hence, we believe that it is of great necessity to understand both a base model’s explainability and also the rationale of its continual learning strategy.

 
 
 
 

## VI Conclusions 

 
 We have presented a comprehensive survey of continual learning (CL) studies relevant to smart city research. We started with the basic task setting and scenarios of CL and then reviewed those widely used CL methods. Next, We categorized the primary applications and learning tasks of CL in smart cities and compiled a list of publicly available datasets commonly used in previous studies. We then offered analysis and examples showcasing the integration of CL with other learning frameworks, such as graph learning, temporal learning, spatial-temporal learning, multi-modality learning, and federated learning. Lastly, we outlined some important challenges faced by CL in smart cities and envisioned future directions. Urban computing and smart city research represent the most complex real-world applications, and we believe that our survey has covered the most fundamental literature on continual learning. We anticipate that this survey could help relevant researchers quickly familiarize themselves with the current state of CL and smart city research and then direct them to future research trends.

 
 
 

## Acknowledgments

 
 This work was supported in part by the National Natural Science Foundation of China (Grant No. 62302405, 62176221, 62276215), the Natural Science Foundation of Sichuan Province (Grant No. 24NSFC2348), China Postdoctoral Science Foundation (Grant No. 2023M732914), Sichuan Science and Technology Program (No. MZGC20230073) and the Fundamental Research Funds for the Central Universities (No. 2682023ZT007)

 
 
 

## References

 
 
 [1] 
 
G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter, “Continual
lifelong learning with neural networks: A review,” Neural
Networks , vol. 113, pp. 54–71, May 2019.

 

 
 [2] 
 
G. M. van de Ven and A. S. Tolias, “Three scenarios for continual learning,”
Apr. 2019.

 

 
 [3] 
 
M. McCloskey and N. J. Cohen, “Catastrophic Interference in
Connectionist Networks: The Sequential Learning Problem,” in
 Psychology of Learning and Motivation . Elsevier, 1989, vol. 24, pp. 109–165.

 

 
 [4] 
 
S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, “iCaRL:
Incremental Classifier and Representation Learning,” in
 Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition . Honolulu,
HI: IEEE, Jul. 2017, pp. 5533–5542.

 

 
 [5] 
 
J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A.
Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis,
C. Clopath, D. Kumaran, and R. Hadsell, “Overcoming catastrophic forgetting
in neural networks,” Proceedings of the National Academy of Sciences ,
vol. 114, no. 13, pp. 3521–3526, Mar. 2017.

 

 
 [6] 
 
Z. Li and D. Hoiem, “Learning without Forgetting,” IEEE
Transactions on Pattern Analysis and Machine Intelligence , vol. 40, no. 12,
pp. 2935–2947, Dec. 2018.

 

 
 [7] 
 
Y. Zheng, L. Capra, O. Wolfson, and H. Yang, “Urban Computing:
Concepts, Methodologies, and Applications,” ACM
Transactions on Intelligent Systems and Technology , vol. 5, no. 3, pp.
1–55, Oct. 2014.

 

 
 [8] 
 
L. Wang, X. Zhang, H. Su, and J. Zhu, “A Comprehensive Survey of
Continual Learning: Theory, Method and Application,” Jan.
2023.

 

 
 [9] 
 
E. Belouadah, A. Popescu, and I. Kanellos, “A comprehensive study of class
incremental learning algorithms for visual tasks,” Neural Networks ,
vol. 135, pp. 38–54, Mar. 2021.

 

 
 [10] 
 
M. De Lange, R. Aljundi, M. Masana, S. Parisot, X. Jia, A. Leonardis,
G. Slabaugh, and T. Tuytelaars, “A continual learning survey: Defying
forgetting in classification tasks,” IEEE Transactions on Pattern
Analysis and Machine Intelligence , vol. 44, no. 7, pp. 3366–3385, Jul.
2022.

 

 
 [11] 
 
Z. Ke and B. Liu, “Continual Learning of Natural Language Processing
Tasks: A Survey,” May 2023.

 

 
 [12] 
 
D.-W. Zhou, Q.-W. Wang, Z.-H. Qi, H.-J. Ye, D.-C. Zhan, and Z. Liu, “Deep
Class-Incremental Learning: A Survey,” Feb. 2023.

 

 
 [13] 
 
Q. Yuan, S.-U. Guan, P. Ni, T. Luo, K. L. Man, P. Wong, and V. Chang,
“Continual Graph Learning: A Survey,” Jan. 2023.

 

 
 [14] 
 
F. G. Febrinanto, F. Xia, K. Moore, C. Thapa, and C. Aggarwal, “Graph
Lifelong Learning: A Survey,” IEEE Computational Intelligence
Magazine , vol. 18, no. 1, pp. 32–51, Feb. 2023.

 

 
 [15] 
 
K. Shaheen, M. A. Hanif, O. Hasan, and M. Shafique, “Continual Learning
for Real-World Autonomous Systems: Algorithms, Challenges and
Frameworks,” Journal of Intelligent Robotic Systems , vol. 105,
no. 1, p. 9, Apr. 2022.

 

 
 [16] 
 
T. Lesort, V. Lomonaco, A. Stoian, D. Maltoni, D. Filliat, and
N. Díaz-Rodríguez, “Continual Learning for Robotics:
Definition, Framework, Learning Strategies, Opportunities and
Challenges,” Nov. 2019.

 

 
 [17] 
 
P. Zhang and S. Kim, “A Survey on Incremental Update for Neural
Recommender Systems,” Mar. 2023.

 

 
 [18] 
 
J. Xiao, Z. Xiao, D. Wang, J. Bai, V. Havyarimana, and F. Zeng, “Short-term
traffic volume prediction by ensemble learning in concept drifting
environments,” Knowledge-Based Systems , vol. 164, pp. 213–225, Jan.
2019.

 

 
 [19] 
 
F. Yu, J. Fang, B. Chen, and Y. Shao, “An Incremental Learning Based
Convolutional Neural Network Model for Large-Scale and Short-Term
Traffic Flow,” International Journal of Machine Learning and
Computing , vol. 11, no. 2, pp. 143–151, Mar. 2021.

 

 
 [20] 
 
Y. Shao, Y. Zhao, F. Yu, H. Zhu, and J. Fang, “The Traffic Flow Prediction
Method Using the Incremental Learning-Based CNN-LTSM Model: The
Solution of Mobile Application,” Mobile Information Systems ,
vol. 2021, p. e5579451, Jun. 2021.

 

 
 [21] 
 
C. Lanza, E. Angelats, M. Miozzo, and P. Dini, “Urban Traffic Forecasting
using Federated and Continual Learning,” in 2023 6th
Conference on Cloud and Internet of Things (CIoT) . Lisbon, Portugal: IEEE, Mar. 2023, pp. 1–8.

 

 
 [22] 
 
X. Chen, J. Wang, and K. Xie, “TrafficStream: A Streaming Traffic Flow
Forecasting Framework Based on Graph Neural Networks and Continual
Learning,” in Proceedings of the Thirtieth International Joint
Conference on Artificial Intelligence (IJCAI-21) . Montreal, Canada: International Joint Conferences on
Artificial Intelligence Organization, Aug. 2021, pp. 3620–3626.

 

 
 [23] 
 
B. Wang, Y. Zhang, J. Shi, P. Wang, X. Wang, L. Bai, and Y. Wang, “Knowledge
Expansion and Consolidation for Continual Traffic Prediction With
Expanding Graphs,” IEEE Transactions on Intelligent Transportation
Systems , pp. 1–12, 2023.

 

 
 [24] 
 
A. Maipradit, Y. Moriyama, T. Okuro, M. Yoshida, N. Tachimori, S. Akiyama,
H. Suwa, and K. Yasumoto, “PAVEMENT: Passing Vehicle Detection
System with Autonomous Incremental Learning using Camera and
Vibration Data,” in 2022 IEEE 96th Vehicular Technology
Conference (VTC2022-Fall) , London, United Kingdom, Sep. 2022, pp.
1–7.

 

 
 [25] 
 
T. Bandaragoda, D. De Silva, D. Kleyko, E. Osipov, U. Wiklund, and
D. Alahakoon, “Trajectory clustering of road traffic in urban environments
using incremental machine learning in combination with hyperdimensional
computing,” in 2019 IEEE Intelligent Transportation Systems
Conference (ITSC) , Auckland, New Zealand, 2019, pp. 1664–1670.

 

 
 [26] 
 
G. Sun, T. Chen, Y. Su, and C. Li, “Internet Traffic Classification Based
on Incremental Support Vector Machines,” Mobile Networks and
Applications , vol. 23, no. 4, pp. 789–796, Aug. 2018.

 

 
 [27] 
 
Z. Wu, R. Qiao, X. Liu, S. Gao, X. Ao, Z. He, and L. Xia, “CEDUP:
Using incremental learning modeling to explore Spatio-temporal carbon
emission distribution and unearthed patterns at the municipal level,”
 Resources, Conservation and Recycling , vol. 193, p. 106980, Jun. 2023.

 

 
 [28] 
 
M. Das, M. Pratama, and S. K. Ghosh, “SARDINE: A Self-Adaptive Recurrent
Deep Incremental Network Model for Spatio-Temporal Prediction of
Remote Sensing Data,” ACM Transactions on Spatial Algorithms and
Systems (TSAS) , vol. 6, no. 3, pp. 16:1–16:26, Apr. 2020.

 

 
 [29] 
 
X. Lu, X. Sun, W. Diao, Y. Feng, P. Wang, and K. Fu, “LIL: Lightweight
Incremental Learning Approach Through Feature Transfer for Remote Sensing
Image Scene Classification,” IEEE Transactions on Geoscience and
Remote Sensing , vol. 60, pp. 1–20, 2022.

 

 
 [30] 
 
X. Wang, L. Yao, X. Wang, H.-Y. Paik, and S. Wang, “Uncertainty Estimation
With Neural Processes for Meta-Continual Learning,” IEEE
Transactions on Neural Networks and Learning Systems , pp. 1–11, 2022.

 

 
 [31] 
 
E. Camargo, J. Aguilar, Y. Quintero, F. Rivas, and D. Ardila, “An incremental
learning approach to prediction models of SEIRD variables in the context
of the COVID-19 pandemic,” Health and Technology , vol. 12, no. 4,
pp. 867–877, Jul. 2022.

 

 
 [32] 
 
C. Hu, Y. Chen, X. Peng, H. Yu, C. Gao, and L. Hu, “A Novel Feature
Incremental Learning Method for Sensor-Based Activity Recognition,”
 IEEE Transactions on Knowledge and Data Engineering , vol. 31, no. 6,
pp. 1038–1050, Jun. 2019.

 

 
 [33] 
 
S. Younan and M. Abu-Elkheir, “Deep Incremental Learning for
Personalized Human Activity Recognition on Edge Devices,” IEEE
Canadian Journal of Electrical and Computer Engineering , vol. 45, no. 3, pp.
215–221, 2022.

 

 
 [34] 
 
V. Gupta, J. Narwariya, P. Malhotra, L. Vig, and G. Shroff, “Continual
Learning for Multivariate Time Series Tasks with Variable Input
Dimensions,” in 2021 IEEE International Conference on Data
Mining (ICDM) , Dec. 2021, pp. 161–170.

 

 
 [35] 
 
S.-Y. Yin, Y. Huang, T.-Y. Chang, S.-F. Chang, and V. S. Tseng, “Continual
learning with attentive recurrent neural networks for temporal data
classification,” Neural Networks , vol. 158, pp. 171–187, Jan. 2023.

 

 
 [36] 
 
A. M. Eldhai, M. Hamdan, S. Khan, M. Hamzah, and M. N. Marsono, “Traffic
Classification based on Incremental Learning Algorithms for the
Software-Defined Networks,” in 2022 International Conference
on Frontiers of Information Technology (FIT) . Islamabad, Pakistan: IEEE, Dec. 2022, pp. 338–343.

 

 
 [37] 
 
Y. Cao, H. Peng, J. Wu, Y. Dou, J. Li, and P. S. Yu, “Knowledge-Preserving
Incremental Social Event Detection via Heterogeneous GNNs,” in
 Proceedings of the Web Conference 2021 , ser. WWW ’21. New York, NY, USA: Association for Computing
Machinery, Jun. 2021, pp. 3383–3395.

 

 
 [38] 
 
P. Yu, H. Ji, and P. Natarajan, “Lifelong Event Detection with Knowledge
Transfer,” in Proceedings of the 2021 Conference on Empirical
Methods in Natural Language Processing . Online and Punta Cana, Dominican Republic: Association for
Computational Linguistics, Nov. 2021, pp. 5278–5290.

 

 
 [39] 
 
H. Peng, R. Zhang, S. Li, Y. Cao, S. Pan, and P. S. Yu, “Reinforced,
Incremental and Cross-Lingual Event Detection From Social Messages,”
 IEEE Transactions on Pattern Analysis and Machine Intelligence ,
vol. 45, no. 1, pp. 980–998, Jan. 2023.

 

 
 [40] 
 
Y. Xu, Y. Zhang, W. Guo, H. Guo, R. Tang, and M. Coates, “GraphSAIL:
Graph Structure Aware Incremental Learning for Recommender Systems,”
in Proceedings of the 29th ACM International Conference on
Information and Knowledge Management (CIKM ’20) , ser. CIKM
’20. Virtual Event, Ireland:
Association for Computing Machinery, Oct. 2020, pp. 2861–2868.

 

 
 [41] 
 
K. Ahrabian, Y. Xu, Y. Zhang, J. Wu, Y. Wang, and M. Coates, “Structure
Aware Experience Replay for Incremental Learning in Graph-based
Recommender Systems,” in Proceedings of the 30th ACM International
Conference on Information and Knowledge Management (CIKM ’21) ,
ser. CIKM ’21. Virtual Event, QLD,
Australia: Association for Computing Machinery, Oct. 2021, pp. 2832–2836.

 

 
 [42] 
 
S. Ding, F. Feng, X. He, Y. Liao, J. Shi, and Y. Zhang, “Causal Incremental
Graph Convolution for Recommender System Retraining,” IEEE
Transactions on Neural Networks and Learning Systems , pp. 1–11, 2022.

 

 
 [43] 
 
H. Amirat, N. Lagraa, P. Fournier-Viger, Y. Ouinten, M. L. Kherfi, and
Y. Guellouma, “Incremental tree-based successive POI recommendation in
location-based social networks,” Applied Intelligence , vol. 53,
no. 7, pp. 7562–7598, Apr. 2023.

 

 
 [44] 
 
Y. Wang, Y. Zhang, and M. Coates, “Graph Structure Aware Contrastive
Knowledge Distillation for Incremental Learning in Recommender
Systems,” in Proceedings of the 30th ACM International
Conference on Information and Knowledge Management (CIKM ’21) ,
ser. CIKM ’21. Virtual Event, QLD,
Australia: Association for Computing Machinery, Oct. 2021, pp. 3518–3522.

 

 
 [45] 
 
J. Xia, D. Li, H. Gu, J. Liu, T. Lu, and N. Gu, “FIRE: Fast Incremental
Recommendation with Graph Signal Processing,” in Proceedings of
the ACM Web Conference 2022 . Virtual Event, Lyon France: ACM, Apr. 2022, pp. 2360–2369.

 

 
 [46] 
 
Z. Cui, X. Sun, L. Pan, S. Liu, and G. Xu, “Event-based incremental
recommendation via factors mixed Hawkes process,” Information
Sciences , vol. 639, p. 119007, Aug. 2023.

 

 
 [47] 
 
H. Ma, Y. Sun, J. Li, M. Tomizuka, and C. Choi, “Continual Multi-Agent
Interaction Behavior Prediction With Conditional Generative Memory,”
 IEEE Robotics and Automation Letters , vol. 6, no. 4, pp. 8410–8417,
Oct. 2021.

 

 
 [48] 
 
Y. Lin, Z. Li, C. Gong, C. Lu, X. Wang, and J. Gong, “Continual Interactive
Behavior Learning With Traffic Divergence Measurement: A Dynamic Gradient
Scenario Memory Approach,” IEEE Transactions on Intelligent
Transportation Systems , pp. 1–18, 2023.

 

 
 [49] 
 
L. Knoedler, C. Salmi, H. Zhu, B. Brito, and J. Alonso-Mora, “Improving
Pedestrian Prediction Models With Self-Supervised Continual Learning,”
 IEEE Robotics and Automation Letters , vol. 7, no. 2, pp. 4781–4788,
Apr. 2022.

 

 
 [50] 
 
B. Yang, F. Fan, R. Ni, J. Li, L. Kiong, and X. Liu, “Continual learning-based
trajectory prediction with memory augmented networks,” Knowledge-Based
Systems , vol. 258, p. 110022, Dec. 2022.

 

 
 [51] 
 
Y. Wu, A. Bighashdel, G. Chen, G. Dubbelman, and P. Jancura, “Continual
Pedestrian Trajectory Learning With Social Generative Replay,”
 IEEE Robotics and Automation Letters , vol. 8, no. 2, pp. 848–855,
Feb. 2023.

 

 
 [52] 
 
D. Gao, C. Wang, and S. Scherer, “AirLoop: Lifelong Loop Closure
Detection,” in 2022 International Conference on Robotics and
Automation (ICRA) , Philadelphia, PA, USA, May 2022, pp.
10 664–10 671.

 

 
 [53] 
 
I. A. Lungu, S.-C. Liu, and T. Delbruck, “Fast event-driven incremental
learning of hand symbols,” in 2019 IEEE International Conference
on Artificial Intelligence Circuits and Systems (AICAS) , Mar.
2019, pp. 25–28.

 

 
 [54] 
 
A. Kanazawa, J. Kinugawa, and K. Kosuge, “Incremental Learning of
Spatial-Temporal Features in Human Motion Patterns with Mixture
Model for Planning Motion of a Collaborative Robot in Assembly
Lines,” in 2019 International Conference on Robotics and
Automation (ICRA) . Montreal,
QC, Canada: IEEE, May 2019, pp. 7858–7864.

 

 
 [55] 
 
R. S. Maharjan, “Continual Learning for Adaptive Affective Human-Robot
Interaction,” in 2022 10th International Conference on
Affective Computing and Intelligent Interaction Workshops and
Demos (ACIIW) . Nara, Japan:
IEEE, Oct. 2022, pp. 1–5.

 

 
 [56] 
 
Y. Cao, M. Jia, P. Ding, X. Zhao, and Y. Ding, “Incremental Learning for
Remaining Useful Life Prediction via Temporal Cascade Broad Learning
System With Newly Acquired Data,” IEEE Transactions on Industrial
Informatics , vol. 19, no. 4, pp. 6234–6245, Apr. 2023.

 

 
 [57] 
 
J. Gao, J. Li, H. Shan, Y. Qu, J. Z. Wang, F.-Y. Wang, and J. Zhang, “Forget
less, count better: A domain-incremental self-distillation learning benchmark
for lifelong crowd counting,” Frontiers of Information Technology 
Electronic Engineering , vol. 24, no. 2, pp. 187–202, Feb. 2023.

 

 
 [58] 
 
L. Zhang, G. Gao, and H. Zhang, “Spatial-Temporal Federated Learning for
Lifelong Person Re-identification on Distributed Edges,” IEEE
Transactions on Circuits and Systems for Video Technology , pp. 1–1, 2023.

 

 
 [59] 
 
X. Rui, Z. Li, Y. Cao, Z. Li, and W. Song, “DILRS: Domain-Incremental
Learning for Semantic Segmentation in Multi-Source Remote Sensing
Data,” Remote Sensing , vol. 15, no. 10, p. 2541, May 2023.

 

 
 [60] 
 
M. Wang, D. Yu, W. He, P. Yue, and Z. Liang, “Domain-incremental learning for
fire detection in space-air-ground integrated observation network,”
 International Journal of Applied Earth Observation and Geoinformation ,
vol. 118, p. 103279, Apr. 2023.

 

 
 [61] 
 
P. Vijayaraghavan and D. Roy, “Lifelong Knowledge-Enriched Social Event
Representation Learning,” in Proceedings of the 16th Conference
of the European Chapter of the Association for Computational
Linguistics: Main Volume . Online: Association for Computational Linguistics, Apr. 2021, pp. 3624–3635.

 

 
 [62] 
 
H. Han, X. Fan, and F. Li, “Prototype Enhancement-Based Incremental
Evolution Learning for Urban Garbage Classification,” IEEE
Transactions on Artificial Intelligence , pp. 1–14, 2023.

 

 
 [63] 
 
Xue Li, Lanshun Nie, Xiandong Si, and Dechen Zhan, “A Class
Incremental Temporal-Spatial Model Based on Wireless Sensor Networks
for Activity Recognition,” in Wireless Algorithms,
Systems, and Applications. WASA 2020 , ser. Lecture Notes in
Computer Science, Dongxiao Yu, Falko Dressler, and Jiguo Yu,
Eds. Cham: Springer International
Publishing, Sep. 2020, pp. 256–271.

 

 
 [64] 
 
N. Veerakumar, J. L. Cremer, and M. Popov, “Dynamic Incremental Learning
for real-time disturbance event classification,” International Journal
of Electrical Power Energy Systems , vol. 148, p. 108988, Jun. 2023.

 

 
 [65] 
 
G. Bovenzi, L. Yang, A. Finamore, G. Aceto, D. Ciuonzo, A. Pescapè, and
D. Rossi, “A First Look at Class Incremental Learning in Deep
Learning Mobile Traffic Classification,” in Proceedings of the 5th
Network Traffic Measurement and Analysis Conference, TMA 2021 ,
Virtual, Sep. 2021.

 

 
 [66] 
 
P. Cao, Y. Chen, J. Zhao, and T. Wang, “Incremental Event Detection via
Knowledge Consolidation Networks,” in Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing
(EMNLP) . Online: Association for
Computational Linguistics, Nov. 2020, pp. 707–717.

 

 
 [67] 
 
M. Liu, S. Chang, and L. Huang, “Incremental Prompting: Episodic Memory
Prompt for Lifelong Event Detection,” in Proceedings of the 29th
International Conference on Computational Linguistics . Gyeongju, Republic of Korea: International
Committee on Computational Linguistics, Oct. 2022, pp. 2157–2165.

 

 
 [68] 
 
R. Aljundi, M. Lin, B. Goujaud, and Y. Bengio, “Gradient based sample
selection for online continual learning,” in Advances in Neural
Information Processing Systems , vol. 32. Curran Associates, Inc., 2019.

 

 
 [69] 
 
D. Nallaperuma, R. Nawaratne, T. Bandaragoda, A. Adikari, S. Nguyen,
T. Kempitiya, D. De Silva, D. Alahakoon, and D. Pothuhera, “Online
Incremental Machine Learning Platform for Big Data-Driven Smart Traffic
Management,” IEEE Transactions on Intelligent Transportation
Systems , vol. 20, no. 12, pp. 4679–4690, 2019.

 

 
 [70] 
 
L. Zhang, G. Gao, and H. Zhang, “Towards Data-Efficient Continuous
Learning for Edge Video Analytics via Smart Caching,” in
 Proceedings of the Twentieth ACM Conference on Embedded Networked
Sensor Systems . Boston
Massachusetts: ACM, Nov. 2022, pp. 1136–1140.

 

 
 [71] 
 
M. Tenzer, Z. Rasheed, and K. Shafique, “Learning citywide patterns of life
from trajectory monitoring,” in Proceedings of the 30th
International Conference on Advances in Geographic Information
Systems , ser. SIGSPATIAL ’22. New York, NY, USA: Association for Computing Machinery, Nov. 2022, pp. 1–12.

 

 
 [72] 
 
J. Xu, J. Zhou, P.-N. Tan, X. Liu, and L. Luo, “Spatio-Temporal Multi-Task
Learning via Tensor Decomposition,” IEEE Transactions on
Knowledge and Data Engineering , vol. 33, no. 6, pp. 2764–2775, Jun. 2021.

 

 
 [73] 
 
J. Farooq and M. A. Bazaz, “A novel adaptive deep learning model of
Covid-19 with focus on mortality reduction strategies,” Chaos,
Solitons Fractals , vol. 138, p. 110148, Sep. 2020.

 

 
 [74] 
 
J. Xiao, L. Chen, H. Chen, and X. Hong, “Baseline Model Training in
Sensor-Based Human Activity Recognition: An Incremental Learning
Approach,” IEEE Access , vol. 9, pp. 70 261–70 272, 2021.

 

 
 [75] 
 
T. Hu, W. Wu, Q. Guo, H. Sun, L. Shi, and X. Shen, “Very short-term spatial
and temporal wind power forecasting: A deep learning approach,”
 CSEE Journal of Power and Energy Systems , vol. 6, no. 2, pp. 434–443,
Jun. 2020.

 

 
 [76] 
 
M. Dubey, P. K. Srijith, and M. S. Desarkar, “Continual Learning for
Time-to-Event Modeling,” in Continual Lifelong Learning
Workshop at ACML 2022 , 2022.

 

 
 [77] 
 
L. Zhao, Y. Gao, J. Ye, F. Chen, Y. Ye, C.-T. Lu, and N. Ramakrishnan,
“Spatio-Temporal Event Forecasting Using Incremental Multi-Source Feature
Learning,” ACM Transactions on Knowledge Discovery from Data ,
vol. 16, no. 2, pp. 40:1–40:28, Sep. 2021.

 

 
 [78] 
 
B. He, X. He, Y. Zhang, R. Tang, and C. Ma, “Dynamically Expandable Graph
Convolution for Streaming Recommendation,” in Proceedings of the
ACM Web Conference 2023 . Austin
TX USA: ACM, Apr. 2023, pp. 1457–1467.

 

 
 [79] 
 
J. Han, H. Liu, S. Liu, X. Chen, N. Tan, H. Chai, and H. Xiong, “iETA: A
Robust and Scalable Incremental Learning Framework for
Time-of-Arrival Estimation,” in Proceedings of the 29th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining , ser.
KDD ’23. New York, NY, USA:
Association for Computing Machinery, Aug. 2023, pp. 4100–4111.

 

 
 [80] 
 
F. Zenke, B. Poole, and S. Ganguli, “Continual Learning Through Synaptic
Intelligence,” in Proceedings of the 34th International
Conference on Machine Learning . PMLR, Jul. 2017, pp. 3987–3995.

 

 
 [81] 
 
R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, and T. Tuytelaars, “Memory
Aware Synapses: Learning what (not) to forget,” in Proceedings
of the European Conference on Computer Vision (ECCV) , 2018, pp.
139–154.

 

 
 [82] 
 
S. Yan, J. Xie, and X. He, “DER: Dynamically Expandable Representation
for Class Incremental Learning,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition . Nashville, TN, USA:
IEEE, Jun. 2021, pp. 3013–3022.

 

 
 [83] 
 
A. Chaudhry, P. K. Dokania, T. Ajanthan, and P. H. S. Torr, “Riemannian
Walk for Incremental Learning: Understanding Forgetting and
Intransigence,” in Computer Vision – ECCV 2018 ,
V. Ferrari, M. Hebert, C. Sminchisescu, and Y. Weiss, Eds. Cham: Springer International Publishing, 2018, vol.
11215, pp. 556–572.

 

 
 [84] 
 
D. Lopez-Paz and M. Ranzato, “Gradient Episodic Memory for Continual
Learning,” in Advances in Neural Information Processing Systems ,
2017, p. 10.

 

 
 [85] 
 
R. J. Hyndman and Y. Khandakar, “Automatic time series forecasting: The
forecast package for R,” Journal of Statistical Software ,
vol. 27, no. 3, pp. 1–22, 2008.

 

 
 [86] 
 
A. J. Smola and B. Schölkopf, “A tutorial on support vector regression,”
 Statistics and Computing , vol. 14, no. 3, pp. 199–222, Aug. 2004.

 

 
 [87] 
 
A. Natekin and A. Knoll, “Gradient boosting machines, a tutorial,”
 Frontiers in Neurorobotics , vol. 7, 2013.

 

 
 [88] 
 
J. Zhang, Y. Zheng, D. Qi, R. Li, and X. Yi, “DNN-based prediction model
for spatio-temporal data,” in Proceedings of the 24th ACM SIGSPATIAL
International Conference on Advances in Geographic Information
Systems . Burlingame California:
ACM, Oct. 2016, pp. 1–4.

 

 
 [89] 
 
Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang, “Graph wavenet for deep
spatial-temporal graph modeling,” in Proceedings of the 28th
International Joint Conference on Artificial Intelligence , ser.
IJCAI’19. Macao, China: AAAI
Press, Aug. 2019, pp. 1907–1913.

 

 
 [90] 
 
B. Yu, H. Yin, and Z. Zhu, “Spatio-temporal graph convolutional networks: A
deep learning framework for traffic forecasting,” in Proceedings of
the 27th International Joint Conference on Artificial Intelligence ,
ser. IJCAI’18. Stockholm, Sweden:
AAAI Press, Jul. 2018, pp. 3634–3640.

 

 
 [91] 
 
I. E. Olatunji and C.-H. Cheng, “Video Analytics for Visual
Surveillance and Applications: An Overview and Survey,” in
 Machine Learning Paradigms: Applications of Learning and
Analytics in Intelligent Systems , ser. Learning and Analytics in
Intelligent Systems, G. A. Tsihrintzis, M. Virvou, E. Sakkopoulos, and
L. C. Jain, Eds. Cham: Springer
International Publishing, 2019, pp. 475–515.

 

 
 [92] 
 
M. Hu, Z. Luo, A. Pasdar, Y. C. Lee, Y. Zhou, and D. Wu, “Edge-Based Video
Analytics: A Survey,” Mar. 2023.

 

 
 [93] 
 
D. Tran, H. Wang, M. Feiszli, and L. Torresani, “Video Classification With
Channel-Separated Convolutional Networks,” in 2019 IEEE/CVF
International Conference on Computer Vision (ICCV) , Oct. 2019, pp.
5551–5560.

 

 
 [94] 
 
Z. Zou, K. Chen, Z. Shi, Y. Guo, and J. Ye, “Object Detection in 20
Years: A Survey,” Proceedings of the IEEE , vol. 111, no. 3,
pp. 257–276, Mar. 2023.

 

 
 [95] 
 
A. Kundu, V. Vineet, and V. Koltun, “Feature Space Optimization for
Semantic Video Segmentation,” in 2016 IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , Jun. 2016, pp.
3168–3175.

 

 
 [96] 
 
H. Zhang, M. Shen, Y. Huang, Y. Wen, Y. Luo, G. Gao, and K. Guan, “A
Serverless Cloud-Fog Platform for DNN-Based Video Analytics with
Incremental Learning,” Feb. 2021.

 

 
 [97] 
 
Romil Bhardwaj, Zhengxu Xia, Ganesh Ananthanarayanan, Junchen Jiang,
Yuanchao Shu, Nikolaos Karianakis, Kevin Hsieh, Paramvir Bahl, and
Ion Stoica, “Ekya: Continuous Learning of Video Analytics Models
on Edge Compute Servers,” in 19th USENIX Symposium on
Networked Systems Design and Implementation (NSDI 22) . Renton, WA: USENIX Association, 2022,apr, pp.
119–135.

 

 
 [98] 
 
Y. Nan, S. Jiang, and M. Li, “Large-scale Video Analytics with
Cloud–Edge Collaborative Continuous Learning,” ACM
Transactions on Sensor Networks , vol. 20, no. 1, pp. 14:1–14:23, Oct. 2023.

 

 
 [99] 
 
G. Wu and S. Gong, “Generalising without Forgetting for Lifelong Person
Re-Identification,” Proceedings of the AAAI Conference on Artificial
Intelligence , vol. 35, no. 4, pp. 2889–2897, May 2021.

 

 
 [100] 
 
N. Pu, W. Chen, Y. Liu, E. M. Bakker, and M. S. Lew, “Lifelong Person
Re-Identification via Adaptive Knowledge Accumulation,” in
 Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) , 2021, pp. 7901–7910.

 

 
 [101] 
 
W. Ge, J. Du, A. Wu, Y. Xian, K. Yan, F. Huang, and W.-S. Zheng, “Lifelong
Person Re-identification by Pseudo Task Knowledge Preservation,”
 Proceedings of the AAAI Conference on Artificial Intelligence ,
vol. 36, no. 1, pp. 688–696, Jun. 2022.

 

 
 [102] 
 
N. Pu, Z. Zhong, N. Sebe, and M. S. Lew, “A Memorizing and Generalizing
Framework for Lifelong Person Re-Identification,” IEEE
Transactions on Pattern Analysis and Machine Intelligence , vol. 45, no. 11,
pp. 13 567–13 585, Nov. 2023.

 

 
 [103] 
 
C. Yu, Y. Shi, Z. Liu, S. Gao, and J. Wang, “Lifelong Person
Re-identification via Knowledge Refreshing and Consolidation,”
 Proceedings of the AAAI Conference on Artificial Intelligence ,
vol. 37, no. 3, pp. 3295–3303, Jun. 2023.

 

 
 [104] 
 
Y. Liu, L. Yao, B. Li, X. Wang, and C. Sammut, “Social Graph Transformer
Networks for Pedestrian Trajectory Prediction in Complex Social
Scenarios,” in Proceedings of the 31st ACM International
Conference on Information Knowledge Management . Atlanta GA USA: ACM, Oct. 2022, pp. 1339–1349.

 

 
 [105] 
 
Y. Lu, W. Wang, X. Hu, P. Xu, S. Zhou, and M. Cai, “Vehicle Trajectory
Prediction in Connected Environments via Heterogeneous Context-Aware
Graph Convolutional Networks,” IEEE Transactions on Intelligent
Transportation Systems , vol. 24, no. 8, pp. 8452–8464, Aug. 2023.

 

 
 [106] 
 
W. Wang, F. Xia, H. Nie, Z. Chen, Z. Gong, X. Kong, and W. Wei, “Vehicle
Trajectory Clustering Based on Dynamic Representation Learning of
Internet of Vehicles,” IEEE Transactions on Intelligent
Transportation Systems , vol. 22, no. 6, pp. 3567–3576, Jun. 2021.

 

 
 [107] 
 
J. Bian, D. Tian, Y. Tang, and D. Tao, “A survey on trajectory clustering
analysis,” Feb. 2018.

 

 
 [108] 
 
A. Rudenko, L. Palmieri, M. Herman, K. M. Kitani, D. M. Gavrila, and K. O.
Arras, “Human motion trajectory prediction: A survey,” The
International Journal of Robotics Research , vol. 39, no. 8, pp. 895–935,
Jul. 2020.

 

 
 [109] 
 
Y. Huang, J. Du, Z. Yang, Z. Zhou, L. Zhang, and H. Chen, “A Survey on
Trajectory-Prediction Methods for Autonomous Driving,” IEEE
Transactions on Intelligent Vehicles , vol. 7, no. 3, pp. 652–674, Sep.
2022.

 

 
 [110] 
 
G. Habibi, N. Jaipuria, and J. P. How, “SILA: An Incremental Learning
Approach for Pedestrian Trajectory Prediction,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) Workshops , 2020, pp. 1024–1025.

 

 
 [111] 
 
D. Zhu, G. Zhai, Y. Di, F. Manhardt, H. Berkemeyer, T. Tran, N. Navab,
F. Tombari, and B. Busam, “IPCC-TP: Utilizing Incremental Pearson
Correlation Coefficient for Joint Multi-Agent Trajectory Prediction,”
in 2023 IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR) , Jun. 2023, pp. 5507–5516.

 

 
 [112] 
 
P. Bao, Z. Chen, J. Wang, D. Dai, and H. Zhao, “Lifelong Vehicle Trajectory
Prediction Framework Based on Generative Replay,” IEEE
Transactions on Intelligent Transportation Systems , vol. 24, no. 12, pp.
13 729–13 741, Dec. 2023.

 

 
 [113] 
 
W. Mao, W. Wang, L. Jiao, S. Zhao, and A. Liu, “Modeling air quality
prediction using a deep learning approach: Method optimization and
evaluation,” Sustainable Cities and Society , vol. 65, p. 102567, Feb.
2021.

 

 
 [114] 
 
X. Yi, J. Zhang, Z. Wang, T. Li, and Y. Zheng, “Deep Distributed Fusion
Network for Air Quality Prediction,” in Proceedings of the 24th
ACM SIGKDD International Conference on Knowledge Discovery Data
Mining . London United Kingdom: ACM,
Jul. 2018, pp. 965–973.

 

 
 [115] 
 
A. Attaallah and R. Ahmad Khan, “SMOTEDNN: A Novel Model for Air
Pollution Forecasting and AQI Classification,” Computers,
Materials Continua , vol. 71, no. 1, pp. 1403–1425, 2022.

 

 
 [116] 
 
W.-L. Mao, W.-C. Chen, C.-T. Wang, and Y.-H. Lin, “Recycling waste
classification using optimized convolutional neural network,”
 Resources, Conservation and Recycling , vol. 164, p. 105132, Jan. 2021.

 

 
 [117] 
 
C. Wu and J. van de Weijer, “Density Map Distillation for Incremental
Object Counting,” in 2023 IEEE/CVF Conference on Computer
Vision and Pattern Recognition Workshops (CVPRW) , Jun. 2023, pp.
2506–2515.

 

 
 [118] 
 
N. Said, K. Ahmad, M. Riegler, K. Pogorelov, L. Hassan, N. Ahmad, and N. Conci,
“Natural disasters detection in social media and satellite imagery: A
survey,” Multimedia Tools and Applications , vol. 78, no. 22, pp.
31 267–31 302, Nov. 2019.

 

 
 [119] 
 
E. Weber, N. Marzo, D. P. Papadopoulos, A. Biswas, A. Lapedriza, F. Ofli,
M. Imran, and A. Torralba, “Detecting Natural Disasters, Damage, and
Incidents in the Wild,” in Computer Vision – ECCV
2020 , ser. Lecture Notes in Computer Science, A. Vedaldi,
H. Bischof, T. Brox, and J.-M. Frahm, Eds. Cham: Springer International Publishing, 2020, pp. 331–350.

 

 
 [120] 
 
N. Churamani, S. Kalkan, and H. Gunes, “Continual Learning for Affective
Robotics: Why, What and How?” in 2020 29th IEEE
International Conference on Robot and Human Interactive
Communication (RO-MAN) . Naples,
Italy: IEEE, Aug. 2020, pp. 425–431.

 

 
 [121] 
 
B. Irfan, A. Ramachandran, S. Spaulding, G. I. Parisi, and H. Gunes, “Lifelong
Learning and Personalization in Long-Term Human-Robot Interaction
(LEAP-HRI),” in 2022 17th ACM/IEEE International
Conference on Human-Robot Interaction (HRI) . Sapporo, Japan: IEEE, Mar. 2022, pp. 1261–1264.

 

 
 [122] 
 
L. Castri, S. Mghames, and N. Bellotto, “From Continual Learning to
Causal Discovery in Robotics,” in Proceedings of The First
AAAI Bridge Program on Continual Causality . PMLR, Jun. 2023, pp. 85–91.

 

 
 [123] 
 
H. Liu, H. Kou, C. Yan, and L. Qi, “Link prediction in paper citation network
to construct paper correlation graph,” EURASIP Journal on Wireless
Communications and Networking , vol. 2019, no. 1, p. 233, Dec. 2019.

 

 
 [124] 
 
C. F. Luo, R. Bhambhoria, S. Dahan, and X. Zhu, “Prototype-Based
Interpretability for Legal Citation Prediction,” in Findings of
the Association for Computational Linguistics: ACL 2023 . Toronto, Canada: Association for
Computational Linguistics, Jul. 2023, pp. 4883–4898.

 

 
 [125] 
 
L. Sun, Z. Zhang, F. Wang, P. Ji, J. Wen, S. Su, and P. S. Yu, “Aligning
Dynamic Social Networks: An Optimization Over Dynamic Graph
Autoencoder,” IEEE Transactions on Knowledge and Data Engineering ,
vol. 35, no. 6, pp. 5597–5611, Jun. 2023.

 

 
 [126] 
 
C. Li, S. Wang, Y. Wang, P. Yu, Y. Liang, Y. Liu, and Z. Li, “Adversarial
Learning for Weakly-Supervised Social Network Alignment,”
 Proceedings of the AAAI Conference on Artificial Intelligence ,
vol. 33, no. 01, pp. 996–1003, Jul. 2019.

 

 
 [127] 
 
L. Zhao, Y. Song, C. Zhang, Y. Liu, P. Wang, T. Lin, M. Deng, and H. Li,
“T-GCN: A Temporal Graph Convolutional Network for Traffic
Prediction,” IEEE Transactions on Intelligent Transportation
Systems , vol. 21, no. 9, pp. 3848–3858, Sep. 2020.

 

 
 [128] 
 
L. Bai, L. Yao, C. Li, X. Wang, and C. Wang, “Adaptive graph convolutional
recurrent network for traffic forecasting,” in Proceedings of the 34th
International Conference on Neural Information Processing Systems ,
ser. NIPS’20. Red Hook, NY, USA:
Curran Associates Inc., Dec. 2020, pp. 17 804–17 815.

 

 
 [129] 
 
S. Guo, Y. Lin, N. Feng, C. Song, and H. Wan, “Attention Based
Spatial-Temporal Graph Convolutional Networks for Traffic Flow
Forecasting,” Proceedings of the AAAI Conference on Artificial
Intelligence , vol. 33, no. 01, pp. 922–929, Jul. 2019.

 

 
 [130] 
 
I. Chami, A. Wolf, D.-C. Juan, F. Sala, S. Ravi, and C. Ré,
“Low-Dimensional Hyperbolic Knowledge Graph Embeddings,” in
 Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics . Online: Association for Computational Linguistics, Jul. 2020, pp. 6901–6914.

 

 
 [131] 
 
K. Zhou, W. X. Zhao, S. Bian, Y. Zhou, J.-R. Wen, and J. Yu, “Improving
Conversational Recommender Systems via Knowledge Graph based
Semantic Fusion,” in Proceedings of the 26th ACM SIGKDD
International Conference on Knowledge Discovery Data Mining ,
ser. KDD ’20. New York, NY, USA:
Association for Computing Machinery, Aug. 2020, pp. 1006–1014.

 

 
 [132] 
 
B. Xue and L. Zou, “Knowledge Graph Quality Management: A Comprehensive
Survey,” IEEE Transactions on Knowledge and Data Engineering ,
vol. 35, no. 5, pp. 4969–4988, May 2023.

 

 
 [133] 
 
W. Ju, Z. Fang, Y. Gu, Z. Liu, Q. Long, Z. Qiao, Y. Qin, J. Shen, F. Sun,
Z. Xiao, J. Yang, J. Yuan, Y. Zhao, X. Luo, and M. Zhang, “A Comprehensive
Survey on Deep Graph Representation Learning,” Apr. 2023.

 

 
 [134] 
 
Y. Ma, Z. Guo, Z. Ren, J. Tang, and D. Yin, “Streaming Graph Neural
Networks,” in Proceedings of the 43rd International ACM SIGIR
Conference on Research and Development in Information Retrieval
(SIGIR ’20) , ser. SIGIR ’20. Virtual Event, China: Association for Computing Machinery, Jul. 2020, pp.
719–728.

 

 
 [135] 
 
Y. Han, S. Karunasekera, and C. Leckie, “Graph Neural Networks with
Continual Learning for Fake News Detection from Social Media,”
Aug. 2020.

 

 
 [136] 
 
A. Daruna, M. Gupta, M. Sridharan, and S. Chernova, “Continual Learning of
Knowledge Graph Embeddings,” IEEE Robotics and Automation
Letters , vol. 6, no. 2, pp. 1128–1135, Apr. 2021.

 

 
 [137] 
 
P. Bielak, K. Tagowski, M. Falkiewicz, T. Kajdanowicz, and N. V. Chawla,
“FILDNE: A Framework for Incremental Learning of Dynamic
Networks Embeddings,” Knowledge-Based Systems , vol. 236, no. C,
Aug. 2021.

 

 
 [138] 
 
J. Wang, G. Song, Y. Wu, and L. Wang, “Streaming Graph Neural Networks via
Continual Learning,” in Proceedings of the 29th ACM
International Conference on Information and Knowledge Management
(CIKM ’20) . Virtual Event,
Ireland: ACM, Oct. 2020, pp. 1515–1524.

 

 
 [139] 
 
J. Wang, W. Zhu, G. Song, and L. Wang, “Streaming Graph Neural Networks
with Generative Replay,” in Proceedings of the 28th ACM SIGKDD
Conference on Knowledge Discovery and Data Mining (KDD ’22) ,
ser. KDD ’22. Washington, DC, USA:
Association for Computing Machinery, Aug. 2022, pp. 1878–1888.

 

 
 [140] 
 
P. Wang, K. Liu, L. Jiang, X. Li, and Y. Fu, “Incremental Mobile User
Profiling: Reinforcement Learning with Spatial Knowledge Graph for
Modeling Event Streams,” in Proceedings of the 26th ACM SIGKDD
Conference on Knowledge Discovery and Data Mining USB Stick
(KDD ’20) , ser. KDD ’20. Virtual Event: Association for Computing Machinery, Aug. 2020, pp. 853–861.

 

 
 [141] 
 
X. Kou, Y. Lin, S. Liu, P. Li, J. Zhou, and Y. Zhang, “Disentangle-based
Continual Graph Representation Learning,” in Proceedings of the
2020 Conference on Empirical Methods in Natural Language
Processing (EMNLP) . Online:
Association for Computational Linguistics, Nov. 2020, pp. 2961–2972.

 

 
 [142] 
 
Z. Yuan, H. Liu, J. Liu, Y. Liu, Y. Yang, R. Hu, and H. Xiong, “Incremental
Spatio-Temporal Graph Learning for Online Query-POI Matching,” in
 Proceedings of the Web Conference 2021 (WWW ’21) , ser. WWW
’21. Ljubljana, Slovenia.: Association
for Computing Machinery, Jun. 2021, pp. 1586–1597.

 

 
 [143] 
 
C. Wang, Y. Qiu, D. Gao, and S. Scherer, “Lifelong Graph Learning,” in
 Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) , 2022, pp. 13 719–13 728.

 

 
 [144] 
 
F. Zhou and C. Cao, “Overcoming Catastrophic Forgetting in Graph Neural
Networks with Experience Replay,” in Proceedings of the AAAI
Conference on Artificial Intelligence , vol. 35, May 2021, pp.
4714–4722.

 

 
 [145] 
 
L. Galke, B. Franke, T. Zielke, and A. Scherp, “Lifelong Learning of
Graph Neural Networks for Open-World Node Classification,” in
 2021 International Joint Conference on Neural Networks
(IJCNN) . Shenzhen, China: IEEE,
Jul. 2021, pp. 1–8.

 

 
 [146] 
 
X. Zhang, D. Song, and D. Tao, “Hierarchical Prototype Networks for
Continual Graph Representation Learning,” IEEE Transactions on
Pattern Analysis and Machine Intelligence , vol. 45, no. 4, pp. 4622–4636,
Apr. 2023.

 

 
 [147] 
 
J. Xia, D. Li, H. Gu, T. Lu, P. Zhang, and N. Gu, “Incremental Graph
Convolutional Network for Collaborative Filtering,” in
 Proceedings of the 30th ACM International Conference on
Information and Knowledge Management (CIKM ’21) , ser. CIKM
’21. Virtual Event, Australia:
Association for Computing Machinery, Oct. 2021, pp. 2170–2179.

 

 
 [148] 
 
B. Lu, X. Gan, L. Yang, W. Zhang, L. Fu, and X. Wang, “Geometer: Graph
Few-Shot Class-Incremental Learning via Prototype Representation,” in
 Proceedings of the 28th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining (KDD ’22) . Washington, DC, USA: ACM, Aug. 2022, pp. 1152–1161.

 

 
 [149] 
 
Z. Tan, K. Ding, R. Guo, and H. Liu, “Graph Few-shot Class-incremental
Learning,” in Proceedings of the Fifteenth ACM International
Conference on Web Search and Data Mining (WSDM ’22) , ser.
WSDM ’22. Virtual Event, Tempe,
AZ, USA: Association for Computing Machinery, Feb. 2022, pp. 987–996.

 

 
 [150] 
 
J. Cai, X. Wang, C. Guan, Y. Tang, J. Xu, B. Zhong, and W. Zhu, “Multimodal
Continual Graph Learning with Neural Architecture Search,” in
 Proceedings of the ACM Web Conference 2022 (WWW ’22) , ser.
WWW ’22. Virtual Event, Lyon,
France: Association for Computing Machinery, Apr. 2022, pp. 1292–1300.

 

 
 [151] 
 
A. Carta, A. Cossu, F. Errica, and D. Bacciu, “Catastrophic Forgetting in
Deep Graph Networks: An Introductory Benchmark for Graph
Classification,” Mar. 2021.

 

 
 [152] 
 
X. Zhang, D. Song, and D. Tao, “CGLB: Benchmark Tasks for Continual
Graph Learning,” in 36th Conference on Neural Information
Processing Systems (NeurIPS 2022) Track on Datasets and
Benchmarks , Sep. 2022.

 

 
 [153] 
 
D. G. Philps, “A Temporal Continual Learning Framework for Investment
Decisions,” Unpublished Doctoral Thesis, City, University of London,
2020.

 

 
 [154] 
 
T. Srinivasan, T.-Y. Chang, L. P. Alva, G. Chochlakis, M. Rostami, and
J. Thomason, “CLiMB: A Continual Learning Benchmark for
Vision-and-Language Tasks,” in Advances in Neural Information
Processing Systems , vol. 35, 2022, pp. 29 440–29 453.

 

 
 [155] 
 
K. Wang, L. Herranz, and J. van de Weijer, “Continual learning in
cross-modal retrieval,” in 2021 IEEE/CVF Conference on
Computer Vision and Pattern Recognition Workshops (CVPRW) , Jun.
2021, pp. 3623–3633.

 

 
 [156] 
 
Y. Gao, N. Fei, H. Lu, Z. Lu, H. Jiang, Y. Li, and Z. Cao, “BMU-MoCo:
Bidirectional momentum update for continual video-language modeling,” in
 Advances in Neural Information Processing Systems , S. Koyejo,
S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp.
22 699–22 712.

 

 
 [157] 
 
X. Zhang, F. Zhang, and C. Xu, “VQACL: A Novel Visual Question Answering
Continual Learning Setting,” in 2023 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR) , Jun. 2023, pp.
19 102–19 112.

 

 
 [158] 
 
Z. Ni, L. Wei, S. Tang, Y. Zhuang, and Q. Tian, “Continual Vision-Language
Representation Learning with Off-Diagonal Information,” in
 Proceedings of the 40th International Conference on Machine
Learning . PMLR, Jul. 2023, pp.
26 129–26 149.

 

 
 [159] 
 
X. Chen, N. Zhang, J. Zhang, X. Wang, T. Wu, X. Chen, Y. Wang, and H. Chen,
“Continual Multimodal Knowledge Graph Construction,” Aug. 2023.

 

 
 [160] 
 
J. Yoon, W. Jeong, G. Lee, E. Yang, and S. J. Hwang, “Federated Continual
Learning with Weighted Inter-client Transfer,” in Proceedings of
the 38th International Conference on Machine Learning . PMLR, Jul. 2021, pp. 12 073–12 086.

 

 
 [161] 
 
Y. Ma, Z. Xie, J. Wang, K. Chen, and L. Shou, “Continual Federated Learning
Based on Knowledge Distillation,” in Proceedings of the
Thirty-First International Joint Conference on Artificial
Intelligence . Honolulu, Hawaii,
USA: International Joint Conferences on Artificial Intelligence Organization,
Jul. 2022, pp. 2182–2188.

 

 
 [162] 
 
D. Shenaj, M. Toldo, A. Rigon, and P. Zanuttigh, “Asynchronous Federated
Continual Learning,” in 2023 IEEE/CVF Conference on
Computer Vision and Pattern Recognition Workshops (CVPRW) , Jun.
2023, pp. 5055–5063.

 

 
 [163] 
 
H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang,
“Informer: Beyond Efficient Transformer for Long Sequence Time-Series
Forecasting,” Proceedings of the AAAI Conference on Artificial
Intelligence , vol. 35, no. 12, pp. 11 106–11 115, May 2021.

 

 
 [164] 
 
J. L. Alcaraz and N. Strodthoff, “Diffusion-based Time Series Imputation
and Forecasting with Structured State Space Models,”
 Transactions on Machine Learning Research , Dec. 2022.

 

 
 [165] 
 
H. Ren, B. Xu, Y. Wang, C. Yi, C. Huang, X. Kou, T. Xing, M. Yang, J. Tong, and
Q. Zhang, “Time-Series Anomaly Detection Service at Microsoft,” in
 Proceedings of the 25th ACM SIGKDD International Conference on
Knowledge Discovery Data Mining . Anchorage AK USA: ACM, Jul. 2019, pp. 3009–3017.

 

 
 [166] 
 
S. Lin, W. Lin, W. Wu, F. Zhao, R. Mo, and H. Zhang, “SegRNN: Segment
Recurrent Neural Network for Long-Term Time Series Forecasting,” Aug.
2023.

 

 
 [167] 
 
S. Siami-Namini, N. Tavakoli, and A. S. Namin, “The Performance of
LSTM and BiLSTM in Forecasting Time Series,” in 2019
IEEE International Conference on Big Data (Big Data) , Dec. 2019,
pp. 3285–3292.

 

 
 [168] 
 
I. E. Livieris, E. Pintelas, and P. Pintelas, “A CNN–LSTM model for
gold price time-series forecasting,” Neural Computing and
Applications , vol. 32, no. 23, pp. 17 351–17 360, Dec. 2020.

 

 
 [169] 
 
P. T. Yamak, L. Yujian, and P. K. Gadosey, “A Comparison between
ARIMA, LSTM, and GRU for Time Series Forecasting,” in
 Proceedings of the 2019 2nd International Conference on
Algorithms, Computing and Artificial Intelligence , ser. ACAI
’19. New York, NY, USA: Association
for Computing Machinery, Feb. 2020, pp. 49–55.

 

 
 [170] 
 
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin, “Attention is All you Need,” in
 Advances in Neural Information Processing Systems , vol. 30. Curran Associates, Inc., 2017.

 

 
 [171] 
 
Q. Wen, T. Zhou, C. Zhang, W. Chen, Z. Ma, J. Yan, and L. Sun, “Transformers
in Time Series: A Survey,” May 2023.

 

 
 [172] 
 
Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long,
“iTransformer: Inverted Transformers Are Effective for Time Series
Forecasting,” Oct. 2023.

 

 
 [173] 
 
A. Das, W. Kong, A. Leach, S. Mathur, R. Sen, and R. Yu, “Long-term
Forecasting with TiDE: Time-series Dense Encoder,” Aug. 2023.

 

 
 [174] 
 
S.-A. Chen, C.-L. Li, N. Yoder, S. O. Arik, and T. Pfister, “TSMixer: An
All-MLP Architecture for Time Series Forecasting,” Sep. 2023.

 

 
 [175] 
 
V. Ekambaram, A. Jati, N. Nguyen, P. Sinthong, and J. Kalagnanam,
“TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series
Forecasting,” in Proceedings of the 29th ACM SIGKDD Conference
on Knowledge Discovery and Data Mining , Aug. 2023, pp. 459–469.

 

 
 [176] 
 
L. Zhao, S. Kong, and Y. Shen, “DoubleAdapt: A Meta-learning Approach
to Incremental Learning for Stock Trend Forecasting,” in
 Proceedings of the 29th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining , ser. KDD ’23. New York, NY, USA: Association for Computing Machinery,
Aug. 2023, pp. 3492–3503.

 

 
 [177] 
 
C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast
adaptation of deep networks,” in Proceedings of the 34th
International Conference on Machine Learning - Volume 70 , ser.
ICML’17. Sydney, NSW, Australia:
JMLR.org, Aug. 2017, pp. 1126–1135.

 

 
 [178] 
 
S. Wang, J. Cao, and P. S. Yu, “Deep Learning for Spatio-Temporal Data
Mining: A Survey,” IEEE Transactions on Knowledge and Data
Engineering , vol. 34, no. 8, pp. 3681–3700, Aug. 2022.

 

 
 [179] 
 
G. Jin, Y. Liang, Y. Fang, J. Huang, J. Zhang, and Y. Zheng,
“Spatio-Temporal Graph Neural Networks for Predictive Learning in
Urban Computing: A Survey,” Apr. 2023.

 

 
 [180] 
 
D. Ramachandram and G. W. Taylor, “Deep Multimodal Learning: A Survey
on Recent Advances and Trends,” IEEE Signal Processing
Magazine , vol. 34, no. 6, pp. 96–108, Nov. 2017.

 

 
 [181] 
 
T. Baltrušaitis, C. Ahuja, and L.-P. Morency, “Multimodal Machine
Learning: A Survey and Taxonomy,” IEEE Transactions on
Pattern Analysis and Machine Intelligence , vol. 41, no. 2, pp. 423–443,
Feb. 2019.

 

 
 [182] 
 
P. P. Liang, A. Zadeh, and L.-P. Morency, “Foundations and Trends in
Multimodal Machine Learning: Principles, Challenges, and Open
Questions,” Feb. 2023.

 

 
 [183] 
 
W. Guo, J. Wang, and S. Wang, “Deep Multimodal Representation Learning:
A Survey,” IEEE Access , vol. 7, pp. 63 373–63 394, 2019.

 

 
 [184] 
 
A. Nagrani, S. Yang, A. Arnab, A. Jansen, C. Schmid, and C. Sun, “Attention
Bottlenecks for Multimodal Fusion,” in Advances in Neural
Information Processing Systems , vol. 34. Curran Associates, Inc., 2021, pp. 14 200–14 213.

 

 
 [185] 
 
H. Luo, L. Ji, B. Shi, H. Huang, N. Duan, T. Li, J. Li, T. Bharti, and M. Zhou,
“UniVL: A Unified Video and Language Pre-Training Model for
Multimodal Understanding and Generation,” Sep. 2020.

 

 
 [186] 
 
H. R. Vaezi Joze, A. Shaban, M. L. Iuzzolino, and K. Koishida, “MMTM:
Multimodal Transfer Module for CNN Fusion,” in 2020
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) , Jun. 2020, pp. 13 286–13 296.

 

 
 [187] 
 
S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and
D. Parikh, “VQA: Visual Question Answering,” in Proceedings
of the IEEE International Conference on Computer Vision (ICCV) ,
2015, pp. 2425–2433.

 

 
 [188] 
 
A. Suhr, S. Zhou, A. Zhang, I. Zhang, H. Bai, and Y. Artzi, “A Corpus for
Reasoning About Natural Language Grounded in Photographs,” Jul.
2019.

 

 
 [189] 
 
S. Goenka, Z. Zheng, A. Jaiswal, R. Chada, Y. Wu, V. Hedau, and P. Natarajan,
“FashionVLP: Vision Language Transformer for Fashion Retrieval
With Feedback,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR) , 2022, pp.
14 105–14 115.

 

 
 [190] 
 
A. Das, S. Kottur, K. Gupta, A. Singh, D. Yadav, J. M. F. Moura, D. Parikh, and
D. Batra, “Visual Dialog,” in Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) ,
2017, pp. 326–335.

 

 
 [191] 
 
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training
of Deep Bidirectional Transformers for Language Understanding,”
 arXiv:1810.04805 [cs] , May 2019.

 

 
 [192] 
 
T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal,
A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M.
Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray,
B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and
D. Amodei, “Language Models are Few-Shot Learners,” Jul. 2020.

 

 
 [193] 
 
A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry,
A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning
Transferable Visual Models From Natural Language Supervision,” Feb.
2021.

 

 
 [194] 
 
K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image
Recognition,” in 2016 IEEE Conference on Computer Vision and
Pattern Recognition (CVPR) , Jun. 2016, pp. 770–778.

 

 
 [195] 
 
A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit,
and N. Houlsby, “An Image is Worth 16x16 Words: Transformers
for Image Recognition at Scale,” 2020.

 

 
 [196] 
 
Y. Zhang, H. Jiang, Y. Miura, C. D. Manning, and C. P. Langlotz, “Contrastive
Learning of Medical Visual Representations from Paired Images and
Text,” 2020.

 

 
 [197] 
 
X. Wang, G. Chen, G. Qian, P. Gao, X.-Y. Wei, Y. Wang, Y. Tian, and W. Gao,
“Large-scale Multi-modal Pre-trained Models: A Comprehensive
Survey,” Machine Intelligence Research , vol. 20, no. 4, pp.
447–482, Aug. 2023.

 

 
 [198] 
 
C. Li, Z. Gan, Z. Yang, J. Yang, L. Li, L. Wang, and J. Gao, “Multimodal
Foundation Models: From Specialists to General-Purpose
Assistants,” Sep. 2023.

 

 
 [199] 
 
S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen, “A Survey on
Multimodal Large Language Models,” Jun. 2023.

 

 
 [200] 
 
J. Konečný, H. B. McMahan, F. X. Yu, P. Richtárik, A. T. Suresh,
and D. Bacon, “Federated Learning: Strategies for Improving
Communication Efficiency,” Oct. 2017.

 

 
 [201] 
 
W. Huang, T. Li, D. Wang, S. Du, J. Zhang, and T. Huang, “Fairness and
accuracy in horizontal federated learning,” Information Sciences ,
vol. 589, pp. 170–185, Apr. 2022.

 

 
 [202] 
 
R. Al-Huthaifi, T. Li, W. Huang, J. Gu, and C. Li, “Federated learning in
smart cities: Privacy and security survey,” Information Sciences ,
vol. 632, pp. 833–857, Jun. 2023.

 

 
 [203] 
 
B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas,
“Communication-Efficient Learning of Deep Networks from
Decentralized Data,” in Proceedings of the 20th International
Conference on Artificial Intelligence and Statistics . PMLR, Apr. 2017, pp. 1273–1282.

 

 
 [204] 
 
N. Shoham, T. Avidor, A. Keren, N. Israel, D. Benditkis, L. Mor-Yosef, and
I. Zeitak, “Overcoming Forgetting in Federated Learning on Non-IID
Data,” Oct. 2019.

 

 
 [205] 
 
Y. Huang, C. Bert, S. Fischer, M. Schmidt, A. Dörfler, A. Maier,
R. Fietkau, and F. Putz, “Continual Learning for Peer-to-Peer
Federated Learning: A Study on Automated Brain Metastasis
Identification,” Nov. 2022.

 

 
 [206] 
 
OpenAI, “GPT-4 Technical Report,” Mar. 2023.

 

 
 [207] 
 
A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham,
H. W. Chung, C. Sutton, S. Gehrmann, P. Schuh, K. Shi, S. Tsvyashchenko,
J. Maynez, A. Rao, P. Barnes, Y. Tay, N. Shazeer, V. Prabhakaran, E. Reif,
N. Du, B. Hutchinson, R. Pope, J. Bradbury, J. Austin, M. Isard,
G. Gur-Ari, P. Yin, T. Duke, A. Levskaya, S. Ghemawat, S. Dev,
H. Michalewski, X. Garcia, V. Misra, K. Robinson, L. Fedus, D. Zhou,
D. Ippolito, D. Luan, H. Lim, B. Zoph, A. Spiridonov, R. Sepassi, D. Dohan,
S. Agrawal, M. Omernick, A. M. Dai, T. S. Pillai, M. Pellat, A. Lewkowycz,
E. Moreira, R. Child, O. Polozov, K. Lee, Z. Zhou, X. Wang, B. Saeta,
M. Diaz, O. Firat, M. Catasta, J. Wei, K. Meier-Hellstern, D. Eck, J. Dean,
S. Petrov, and N. Fiedel, “PaLM: Scaling Language Modeling with
Pathways,” Oct. 2022.

 

 
 [208] 
 
H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix,
B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin,
E. Grave, and G. Lample, “LLaMA: Open and Efficient Foundation
Language Models,” Feb. 2023.

 

 
 [209] 
 
Z. Ke, H. Lin, Y. Shao, H. Xu, L. Shu, and B. Liu, “Continual Training of
Language Models for Few-Shot Learning,” in Proceedings of the
2022 Conference on Empirical Methods in Natural Language
Processing . Abu Dhabi, United Arab
Emirates: Association for Computational Linguistics, Dec. 2022, pp.
10 205–10 216.

 

 
 [210] 
 
J. Jang, S. Ye, S. Yang, J. Shin, J. Han, G. Kim, S. J. Choi, and M. Seo,
“Towards Continual Knowledge Learning of Language Models,” in
 The Tenth International Conference on Learning
Representations , Virtual Event, 2022-04-25/2022-04-29.

 

 
 [211] 
 
Z. Ke, Y. Shao, H. Lin, T. Konishi, G. Kim, and B. Liu, “CONTINUAL
PRE-TRAINING OF LANGUAGE MODELS,” in The Eleventh International
Conference on Learning Representations, ICLR 2023 . OpenReview.net, 2023.

 

 
 [212] 
 
A. Razdaibiedina, Y. Mao, R. Hou, M. Khabsa, M. Lewis, and A. Almahairi,
“Progressive Prompts: Continual Learning for Language Models,”
in The Eleventh International Conference on Learning
Representations, { } ICLR { } , Kigali, Rwanda,
2023-05-01/2023-05-05.

 

 
 [213] 
 
T. Wu, L. Luo, Y.-F. Li, S. Pan, T.-T. Vu, and G. Haffari, “Continual
Learning for Large Language Models: A Survey,” Feb. 2024.

 

 
 [214] 
 
J. Li, D. Li, S. Savarese, and S. Hoi, “BLIP-2: Bootstrapping
Language-Image Pre-training with Frozen Image Encoders and Large
Language Models,” Jun. 2023.

 

 
 [215] 
 
B. Li, Y. Zhang, L. Chen, J. Wang, F. Pu, J. Yang, C. Li, and Z. Liu,
“MIMIC-IT: Multi-Modal In-Context Instruction Tuning,” Jun. 2023.

 

 
 [216] 
 
Y. Shen, K. Song, X. Tan, D. Li, W. Lu, and Y. Zhuang, “HuggingGPT:
Solving AI Tasks with ChatGPT and its Friends in Hugging
Face,” May 2023.

 

 
 [217] 
 
C. Wu, S. Yin, W. Qi, X. Wang, Z. Tang, and N. Duan, “Visual ChatGPT:
Talking, Drawing and Editing with Visual Foundation Models,”
Mar. 2023.

 

 
 [218] 
 
A. Awadalla, I. Gao, J. Gardner, J. Hessel, Y. Hanafy, W. Zhu, K. Marathe,
Y. Bitton, S. Gadre, S. Sagawa, J. Jitsev, S. Kornblith, P. W. Koh,
G. Ilharco, M. Wortsman, and L. Schmidt, “OpenFlamingo: An Open-Source
Framework for Training Large Autoregressive Vision-Language Models,”
Aug. 2023.

 

 
 [219] 
 
Y. Huang, K. Xie, H. Bharadhwaj, and F. Shkurti, “Continual Model-Based
Reinforcement Learning with Hypernetworks,” in 2021 IEEE
International Conference on Robotics and Automation (ICRA) ,
May 2021, pp. 799–805.

 

 
 [220] 
 
M. Wolczyk, M. Zajac, R. Pascanu, L. Kucinski, and P. Milos, “Continual
World: A Robotic Benchmark For Continual Reinforcement Learning,” in
 Advances in Neural Information Processing Systems , vol. 34. Curran Associates, Inc., 2021, pp.
28 496–28 510.

 

 
 [221] 
 
K. Khetarpal, M. Riemer, I. Rish, and D. Precup, “Towards Continual
Reinforcement Learning: A Review and Perspectives,” Journal
of Artificial Intelligence Research , vol. 75, pp. 1401–1476, Dec. 2022.

 

 
 [222] 
 
D. Abel, A. Barreto, B. V. Roy, D. Precup, H. van Hasselt, and S. Singh, “A
Definition of Continual Reinforcement Learning,” in
 Thirty-Seventh Conference on Neural Information Processing
Systems , Nov. 2023.

 

 
 [223] 
 
B. Liu, S. Mazumder, E. Robertson, and S. Grigsby, “AI Autonomy:
Self-initiated Open-world Continual Learning and Adaptation,”
 AI Magazine , vol. 44, no. 2, pp. 185–199, 2023.

 

 
 [224] 
 
T.-D. Truong, H.-Q. Nguyen, B. Raj, and K. Luu, “Fairness Continual Learning
Approach to Semantic Scene Understanding in Open-World
Environments,” in Thirty-Seventh Conference on Neural
Information Processing Systems , Nov. 2023.

 

 
 [225] 
 
G. Kim, C. Xiao, T. Konishi, Z. Ke, and B. Liu, “Open-World Continual
Learning: Unifying Novelty Detection and Continual Learning,” Apr.
2023.

 

 
 [226] 
 
Y. Li, X. Yang, H. Wang, X. Wang, and T. Li, “Learning to Prompt Knowledge
Transfer for Open-World Continual Learning,” Proceedings of the
AAAI Conference on Artificial Intelligence , vol. 38, no. 12, pp.
13 700–13 708, Mar. 2024.

 

 
 [227] 
 
D. Li, N. Huang, Z. Wang, and H. Yang, “Personalized Federated Continual
Learning for Task-incremental Biometrics,” IEEE Internet of
Things Journal , pp. 1–1, 2023.

 

 
 [228] 
 
V. De Caro, C. Gallicchio, and D. Bacciu, “Continual adaptation of federated
reservoirs in pervasive environments,” Neurocomputing , vol. 556, p.
126638, Nov. 2023.

 

 
 [229] 
 
X. Qi, Y. Zeng, T. Xie, P.-Y. Chen, R. Jia, P. Mittal, and P. Henderson,
“Fine-tuning Aligned Language Models Compromises Safety, Even When
Users Do Not Intend To!” in The Twelfth International Conference
on Learning Representations , Oct. 2023.

 

 
 [230] 
 
A. R. Javed, W. Ahmed, S. Pandya, P. K. R. Maddikunta, M. Alazab, and T. R.
Gadekallu, “A Survey of Explainable Artificial Intelligence for
Smart Cities,” Electronics , vol. 12, no. 4, p. 1020, Jan. 2023.

 

 
 [231] 
 
K. Ahmad, M. Maabreh, M. Ghaly, K. Khan, J. Qadir, and A. Al-Fuqaha,
“Developing future human-centered smart cities: Critical analysis of
smart city security, Data management, and Ethical challenges,”
 Computer Science Review , vol. 43, p. 100452, Feb. 2022.

 

 
 [232] 
 
D. Rymarczyk, J. van de Weijer, B. Zieliński, and B. Twardowski,
“ICICLE: Interpretable Class Incremental Continual Learning,” Jul.
2023.