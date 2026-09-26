Information Cascade Prediction under Public Emergencies: A Survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2404.01319v2 [cs.SI] 16 May 2024 
 
 

# Information Cascade Prediction under Public Emergencies: A Survey

 CCS:  Applied computing Forecasting CCS:  Applied computing Decision analysis CCS:  Information systems Information integration 
 
 
 Qi Zhang
 
 
 
 email: 230228520@seu.edu.cn 
 
 Affiliation:  southeast university , NanJing , JiangSU , China , 211189 
 
 , 
 Guang Wang
 
 email: guang@cs.fsu.edu 
 
 Affiliation:  Florida State University , USA 
 
 , 
 Li Lin
 
 email: linli321@seu.edu.cn 
 
 Affiliation:  southeast university , NanJing , JiangSU , China , 211189 
 
 , 
 Kaiwen Xia
 
 email: 230228518@seu.edu.cn 
 
 Affiliation:  southeast university , NanJing , JiangSU , China , 211189 
 
 and 
 Shuai Wang
 
 email: shuaiwang@seu.edu.cn 
 
 Affiliation:  southeast university , NanJing , JiangSU , China , 211189 
 

 Abstract. 
 
 With the advent of the era of big data, massive information, expert experience, and high-accuracy models bring great opportunities to the information cascade prediction of public emergencies. However, the involvement of specialist knowledge from various disciplines has resulted in a primarily application-specific focus (e.g., earthquakes, floods, infectious diseases) for information cascade prediction of public emergencies. The lack of a unified prediction framework poses a challenge for classifying intersectional prediction methods across different application fields. This survey paper offers a systematic classification and summary of information cascade modeling, prediction, and application. We aim to help researchers identify cutting-edge research and comprehend models and methods of information cascade prediction under public emergencies. By summarizing open issues and outlining future directions in this field, this paper has the potential to be a valuable resource for researchers conducting further studies on predicting information cascades.

 
 
 
 Keywords:  Public emergencies, Information cascade, Risk prediction, Vulnerability prediction
 
 

## 1. Introduction

 

### 1.1. Background

 
 Information Cascade Prediction is a critical research field that has implications in a variety of domains, such as healthcare, business, cyber domain, politics, and entertainment, ultimately impacting nearly every aspect of our lives ( Sun et al., 2020 ) . These emergencies are unexpected events that occur suddenly and result in or have the potential to result in significant casualties, property damage, ecological harm, and serious social consequences ( Ogie et al., 2018 ) . Throughout history, natural disasters (such as earthquakes, tsunamis, volcanic eruptions, storms, floods, avalanches, droughts, and wildfires) and accident disasters (including environmental disasters, traffic accidents, explosions, and gas leaks) have caused numerous fatalities, infrastructure damage, and extensive economic loss. According to the Emergencies Database (EM-DAT), between 2000 and 2023, 5,922 public emergencies occurred, leading to 480,000 casualties and 3.5 trillion in economic losses, as shown in Figure 1 ( emd, 2023 ) . Therefore, it is increasingly vital to use data, information, and various models to predict potential public emergencies that jeopardize public safety and well-being. Predicting the cascade of information in the event deduction process under public emergencies assists governments, organizations, and individuals in taking proactive measures to mitigate the impact of emergencies and minimize damage.

 
 
 Public emergencies are classified into different categories. The most common categories of public emergencies include (1) Natural disasters , (2) Accident disasters .

 
 
 
 
 
 (a) Global Distribution from Natural Disasters, 2000 to 2023 
 
 
 (b) Global Distribution from Technological Disasters, 2000 to 2023 
 
 Figure 1. Global Distribution of Public Emergencies 
 
 

#### 1.1.1. Natural Disaster

 
 Natural disasters pose a significant and recurring threat to human populations and infrastructure worldwide. These disasters range from sudden, catastrophic events such as volcanic eruptions, earthquakes, floods, hurricanes, and storms to slower, gradual processes like land desertification, soil erosion, environmental degradation, and major infectious disease outbreaks, mass unexplained diseases. For instance, the COVID-19 pandemic, which began in December 2019, has had a devastating impact globally, with over 6.5 million deaths reported to the World Health Organization as of 21 October 2022, and more than 623 million confirmed cases. In addition to the immediate impact of natural disasters, derivative disasters often follow, which exacerbate the initial damage caused by the event. Therefore, it is crucial to recognize the complex and interconnected nature of natural disasters, as one disaster trigger or contribute to several others.

 
 
 

#### 1.1.2. Accident Disaster

 
 Accident Disaster is a term that refers to unexpected events occurring during people’s production or daily life, typically caused by human activities, and resulting in a substantial number of casualties, economic losses, or environmental pollution. Examples of Accident Disasters include fires, explosions, and toxic leaks during industrial production; collapses, gas and coal dust explosions, and flooding during mining; and shipwrecks, major traffic accidents, and aircraft accidents during river-sea transportation. Accident Disasters are a type of public emergency with universality, randomness, inevitability, causal correlation, mutation, latency, and harmfulness. Disasters significantly impact people’s lives and production, resulting in severe social consequences.

 
 
 
 

### 1.2. Information Cascade Prediction under Public Emergencies

 
 Public emergencies often create uncertainty and panic, leading individuals to be influenced by the behavior of those around them. The phenomenon where people rely on the actions of others rather than their own judgment or the development of events to form impressions, often due to external factors, is referred to as information cascades. The cascading information predicts events, anticipate their progress and human behavior, and predicts their impact on other events or sub-events. Information Cascade Prediction involves analyzing and predicting how information spreads and affects decision-making during crises or emergencies. Researchers and practitioners utilize diverse algorithms to predict and manage information cascades during public emergencies.

 
 
 Decision Trees are a type of supervised learning algorithm that works by recursively partitioning the data based on various features or parameters until a stopping criterion is met. Bayesian Networks are probabilistic graphical models that represent and reason about uncertain relationships between various factors. They are used to predict the likelihood of cascading events or hazards by incorporating various factors and their probabilistic relationships.

 
 
 Matrix Factorization algorithms, such as Singular Value Decomposition and Non-negative Matrix Factorization, are effective tools for analyzing and predicting the spread of a pandemic or other public health crises. Support Vector Machines are commonly used to predict the path and intensity of natural disasters like hurricanes or typhoons, taking factors like wind speed and atmospheric pressure into account. Similarly, Random Forests predict the likelihood of cascading failures during a cyber attack on critical infrastructure by analyzing various factors related to the attack and the resilience of the infrastructure.

 
 
 Neural Networks, including Long Short-Term Memory Networks (LSTMs) and Graph Neural Networks (GNNs), are capable of learning complex patterns and relationships in data, which leads to improved prediction accuracy. LSTMs predict future actions based on past behavior patterns, while GNNs incorporate social network data to predict the spread of behavior patterns. Other algorithms like Deep Learning models, Generative Adversarial Networks (GANs), Autoencoders, Attention mechanisms, and Transformer models are also valuable for learning complex patterns and relationships in data and improving prediction accuracy. Each of these algorithms is used to predict various events and hazards using different types of data and relationships between them.

 
 
 In summary, the choice of algorithm for information cascade prediction in public emergencies depends on the specific task and the level of complexity required. Each algorithm has its strengths and limitations, and the appropriate choice depends on the nature of the problem and the available data.

 
 
 

### 1.3. Summary of Existing Surveys

 
 It is well-known that data-driven approaches are commonly applied at 4 stages of public emergencies, including mitigation, preparedness, response, and recovery. However, predicting public emergencies is challenging due to the uncertain characteristics of events. In recent years, the rapid development of machine learning methods has led to significant progress in the development and application of information cascade prediction technology. The various data-driven methods are applied at each stage of a public emergency as shown in Table 1 .

 
 
 Table 1. Compare With Existing Survey 
 
 
 Surveys | 
 Regular Events | 
 Public Emergency | 

 
 | 
 | 
 Model | 
 Prediction | 
 Multi—stage Application | 

 
 | 
 | 
 | 
 Direct Risk | 
 Sequence Risk | 
 Cascading Risk | 
 Before | 
 During | 
 After | 

 
 ( Gao et al., 2019a ) | 
 ✓ | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 ( Duan et al., 2020 ) | 
 ✓ | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 ( Moniz and Torgo, 2019 ) | 
 ✓ | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 ( Tatar et al., 2014 ) | 
 ✓ | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 ( Szabo and Huberman, 2010 ) | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 | 

 
 ( Yang and Counts, 2010 ) | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 ✓ | 
 | 

 
 ( Coglianese and Nash, 2016 ) | 
 | 
 | 
 ✓ | 
 | 
 | 
 | 
 ✓ | 
 | 

 
 ( Blair and Sambanis, 2020 ) | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 | 
 | 

 
 Our Work | 
 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 

 
 
 In recent years, several studies ( Blair and Sambanis, 2020 ; Coglianese and Nash, 2016 ; Duan et al., 2020 ; Gao et al., 2019a ; Moniz and Torgo, 2019 ; Szabo and Huberman, 2010 ; Tatar et al., 2014 ; Yang and Counts, 2010 ) have reviewed prediction algorithms for public emergencies. However, none of these studies have presented a comprehensive overview of methods that effectively incorporate information cascade prediction modeling, prediction, and application. Gao and Duan et al. ( Duan et al., 2020 ; Gao et al., 2019a ) have emphasized event modeling and simple risk forecasting using 1D structured data. Nuno, Alexandru, and Gabor et al. ( Moniz and Torgo, 2019 ; Szabo and Huberman, 2010 ; Tatar et al., 2014 ; Yang and Counts, 2010 ) have focused on summarizing information cascade prediction methods under public emergencies and analyzing the application of various feature engineering methods. Ali and Robert et al. ( Blair and Sambanis, 2020 ; Coglianese and Nash, 2016 ) have considered information cascade prediction for emergency public events and summarized application methods for different stages of emergency events. Although these reviews have approached public emergency prediction methods from different perspectives, they fail to consider the influence of each stage of a public emergency and the event evolution caused by the information cascade, as well as time, place, semantics, deduction, and other features.

 
 
 To address this gap, our work focuses on developing systematic and standardized summary methods to evaluate various information cascade prediction methods. These efforts help identify bottlenecks, pitfalls, open problems, and potentially fruitful future research directions.

 
 
 Figure 2. The Overall Structure Of the Survey 
 
 
 

### 1.4. Contributions of This Paper

 
 The motivation behind conducting a comprehensive review of the field of information cascade prediction is to gain a deeper understanding of the methods and techniques used to predict information cascades and analyze the diffusion of individuals. By examining current research in these areas, we identify gaps in knowledge, and potential areas for improvement, and determine the most effective approaches and people flow at different levels of interaction. Different from previous surveys, our work focuses on information cascade prediction methods under emergencies. This review aims to provide a comprehensive overview of the field, highlighting key findings and advancements in information cascade prediction and people flow analysis. Compared to previous investigations, this paper makes several unique contributions.

 
 
 
 ∙ \bullet 
 
 Firstly, it systematically classifies and summarizes existing technologies, categorizing them according to event aspects, problem formulation, and corresponding techniques. This taxonomy helps domain experts find the most useful techniques for their target problem settings.

 

 ∙ \bullet 
 
 Secondly, the paper conducts a finer-grained analysis of existing information cascade prediction methods for emergencies. It also considers the information cascades and prevalence prediction of emergencies, explaining and analyzing their characteristics and methods.

 

 ∙ \bullet 
 
 Thirdly, the paper provides a comprehensive and up-to-date literature review of information cascade prediction methods, covering both the macro-model of collective behavior and the micro-model of individual user responses. These approaches incorporate recent advances in modeling and predicting information popularity.

 

 ∙ \bullet 
 
 Lastly, the paper discusses the research status and future trends in the field, outlining the overall situation and shape of the current research front. It concludes with new insights into bottlenecks, pitfalls, and open issues, and discusses possible future directions.

 

 
 
 
 

### 1.5. Roadmap

 
 The remainder of this paper is organized as shown in Figure 2 . Section 2 presents Problem Formulation and Performance Evaluation. Section 3 introduces the Information Cascade Model under Public Emergencies. Section 4 introduces the Deduction of Public Emergencies by joint information cascade prediction, including joint time and semantics (Section 4.1), joint time and location (Section 4.2), joint time, location, and semantics (Section 4.3), Vulnerability Prediction (Section 4.4), Association-based Impact Prediction (Section 4.5), and Causality-based Prediction (Section 4.6). Section 5 introduces the Application to Public Emergencies and points out directions for future work, and the survey concludes in Section 5.

 
 
 
 

## 2. Definition of Information Cascade Prediction

 
 Due to the particularity of public emergencies, there are usually information cascade items between public emergencies. This information is usually heterogeneous, sparse, and biased data due to different acquisition channels and methods. This section introduces the formulas and classifications used in the information cascade prediction process of public emergencies: (1) Information Cascade ; (2) Prediction .

 
 

### 2.1. Information Cascade

 
 Information-level correlation in public emergencies often refers to the information correlation between sub-events or between events and events in public emergencies. The information cascade prediction of public emergencies is essentially a regression problem. The development trend of public emergencies is represented by information diffusion.

 
 
 Definition 2.1 (Information Cascade) Assume that there are M M information items I 1 , I 2 , … , I M {I_{1},I_{2},...,I_{M}} for event y y . I i I_{i} represents the cascading degree of events at time t t , that is, the probability that event y y may affect subsequent events. and information cascade prediction aims to predict the occurrence probability P i ​ ( t ) P_{i}(t) at time t t in the future.

 
 
 Information Cascade Prediction aims to predict whether a cascading event occurs given a predefined absolute/relative threshold. For example, whether the event occurs ( Naveed et al., 2011 ; Petrovic et al., 2011 ) , and whether the public emergency causes other cascading events in the future ( Cheng et al., 2014 ; Cui et al., 2013 ) , in addition, the classification task is also used to predict the probability of the occurrence of the emergency. It is possible to fall into which interval ( Gao et al., 2014b ; Hong et al., 2011 ; Ma et al., 2013 ; Ma et al., 2012 ) , which is the classic multi-class classification task. Since the classification task is similar to the low-level version of the regression task, it has lower dimensions in the data parameters, making it easier to achieve good results. Developing regression strategies relies on expert knowledge and requires a fine-grained scope to analyze which factors affect predicting cascading events in public emergencies. Due to the complexity of cascades among public emergencies, accurate regression predictions usually require more information about events as well as cascades ( Gao et al., 2014a ) . Furthermore, it suffers from undesirable problems such as overfitting, inductive bias, and accumulation of prediction errors ( Xiao et al., 2016 ) .

 
 
 

### 2.2. Prediction

 
 Public emergencies contain three important information items: time, location, and semantic ( Shao et al., 2017 ) . Let sudden public emergencies be represented by y = ( t , l , s ) y=(t,l,s) , t ∈ T t\in T represents the time factor, l ∈ L l\in L represents the positioning element, and s ∈ S s\in S represents the semantic element. T T , L L , and S represent the time, location, and semantic domains. Among them, the semantic domain contains any type of semantic feature useful in detailing the semantics of various aspects of an event, including its actors, objects, actions, size, textual description, and other analytical information.

 
 
 In the context of predicting information cascades during public emergencies, it is possible to independently predict the time, location, and semantics of disasters while predicting other factors of the emergency through different elements of the three. Alternatively, the cascading nature of public emergencies is used to predict the occurrence of public emergencies, with precursor events and subsequent events being jointly predicted. The emergency information is represented by X ⊆ T ∗ L ∗ F X\subseteq T*L*F , and F represents the set of semantic information. The current time is expressed as t t , past time and future time are expressed as T − ≡ { t | t n ​ o ​ w ≤ t , t ∈ T } T^{-}\equiv\{t|t_{now}\leq t,t\in T\} and T + ≡ { t | t t n ​ o ​ w , t ∈ T } T^{+}\equiv\{t|t t_{now},t\in T\} , the public emergencies formula is as follows:

 
 
 Definition 2.2 (Public Emergency Prediction) Let the set of precursory events be expressed as X ⊆ T − ∗ L ∗ S X\subseteq T^{-}*L*S , and the data of subsequent precursory events be expressed as Y 0 ⊆ T − ∗ L ∗ S Y_{0}\subseteq T^{-}*L*S . Based on the information of precursor events and subsequent events, the information of public emergencies is expressed as Y ^ ⊆ T − ∗ L ∗ S \hat{Y}\subseteq T^{-}*L*S . So that each future emergency is expressed as y ^ = ( t , l , s ) ∈ Y ^ \hat{y}=(t,l,s)\in\hat{Y} where t t n ​ o ​ w t t_{now} .

 
 
 
 

## 3. Information Cascade Model

 
 The Information Cascade Model is a predictive model that utilizes feature extraction techniques to predict the spread and severity of public emergencies. This model is designed to identify relevant factors that influence the occurrence and spread of emergencies and to incorporate these factors into a machine learning framework for predictive modeling. The Information Cascade Model incorporates various types of features, including Temporal Features , Structural Features , User/Ttem Features , and Content Features , to capture different aspects of public emergency occurrences. Temporal features capture changes in emergency occurrences over time, while structural features capture relationships between emergency response networks. User/item features capture relevant demographic and socioeconomic factors, while content features capture emergency data.

 
 

### 3.1. Temporal Features

 
 Temporal features play a crucial role in predicting the spread of information items or cascades during public emergencies. In this section, we discuss the importance of different types of temporal features in feature engineering for prediction. Specifically, we focus on four subtopics: Observation Time, Occurrence Time, Participation Time, and Evolving Trends as shown in Table 2 .

 
 
 Table 2. Temporal and Structural Features of Public Emergencies 
 
 
 
 
 Feature 
 | 
 
 
 Definition 
 | 
 
 
 References 
 | 

 
 
 
 Observation Time 
 | 
 
 
 The observation time of early features and precursor events 
 | 
 
 
 ( Yang and Leskovec, 2011 ; Cheng et al., 2014 ; Szabo and Huberman, 2010 ) 
 | 

 
 
 
 Occurrence Time 
 | 
 
 
 The time at which a public emergency begins 
 | 
 
 
 ( Lakkaraju et al., 2013 ; Matsubara et al., 2012b ; Szabo and Huberman, 2010 ; Tatar et al., 2011 ; Gao et al., 2014b ; Wu et al., 2016 ) 
 | 

 
 
 
 Participation Time 
 | 
 
 
 The time at which an individual participates in a public emergency 
 | 
 
 
 ( Barabasi, 2005 ; Tsagkias et al., 2010 ; Petrovic et al., 2011 ) 
 | 

 
 
 
 Evolving Trends 
 | 
 
 
 The Developments in public opinion, behavior, or Events over Time 
 | 
 
 
 ( Gürsun et al., 2011 ; Leskovec et al., 2009 ; Yang and Leskovec, 2011 ) 
 | 

 
 
 
 Cascade Graph 
 | 
 
 
 A directed graph representing the spread of information from an initial node 
 | 
 
 
 ( Pillai et al., 2016 ; Asim et al., 2016 ; Caigny et al., 2020 ; Zhao et al., 2016 ) 
 | 

 
 
 
 Global Graph 
 | 
 
 
 A fundamental representation of the relationships between nodes 
 | 
 
 
 ( Chan and Franklin, 2011 ; Decroos et al., 2017 ; Deep et al., 2020 ; Ding et al., 2018 ; ElRefai et al., 2022 ; Allison, 2018 ; Gallego-Castillo et al., 2015 ; Gao and Zhao, 2018 ; Laxman et al., 2008 ; Minor and Cook, 2017 ; Piraján et al., 2019 ; Povinelli and Feng, 2003 ; Vahedian et al., 2017 ) 
 | 

 
 
 
 r-reachable Graph 
 | 
 
 
 A sub-graph extracted from the global graph based on cascade nodes 
 | 
 
 
 ( Deep et al., 2020 ; Lin et al., 2018 ) 
 | 

 
 

#### 3.1.1. Observation Time

 
 Temporal features play a critical role in predicting the popularity of public emergencies. These features are typically extracted based on the peeking strategy, which involves observing a small number of early features and their action time to obtain a sequence of timestamps that are utilized for feature selection. However, the length of the time series is highly irregular, and directly utilizing timestamps as a feature is often ineffective in practice. To address this, transformations are often applied in advance ( Yang and Leskovec, 2011 ) . For example, the period may be divided into evenly distributed intervals, and the cumulative or incremental popularity may be calculated, or a fixed number of early features may be observed ( Cheng et al., 2014 ) . To predict the popularity P i ​ ( t p ) P_{i}(t_{p}) at prediction time t p t_{p} based on the information observed at time t o t_{o} , previous studies have analyzed the relationships between the log-transformed popularity P i ​ ( t p ) P_{i}(t_{p}) and P i ​ ( t o ) P_{i}(t_{o}) . One such study ( Szabo and Huberman, 2010 ) found a high correlation between early-stage and future popularity and used a simple linear prediction model that takes the early observed popularity as input to predict future popularity.

 
 
 

#### 3.1.2. Occurrence Time

 
 Temporal features are crucial in predicting the popularity of public emergencies, with occurrence time ( t 0 t_{0} ) being a particularly important factor. Previous studies ( Lakkaraju et al., 2013 ; Matsubara et al., 2012b ; Szabo and Huberman, 2010 ; Tatar et al., 2011 ; Tsagkias et al., 2010 ) have highlighted the strong relationship between the popularity of public emergencies and the time they occurred, with items posted during the day generally more popular than those posted at midnight. To overcome the effect of user activity periodicity, researchers have proposed various solutions. For instance, local models were designed with each model trained on samples published at a specific hour during the day ( Petrovic et al., 2011 ) . Tweet time was utilized to eliminate the imbalanced diurnal effect of user activities ( Gao et al., 2014b ) . Other temporal factors, such as source time ( Tsagkias et al., 2010 ) and activeness variability ( Wu et al., 2016 ) , have also been employed to improve the robustness of prediction models. However, to improve the accuracy of prediction models, other temporal features, such as the age of the tweet and the time since the event started, should also be considered. For example, a study ( Wu et al., 2019 ) showed that the age of a tweet has a negative correlation with its popularity, indicating that tweets posted early during an event are more likely to be popular than those posted later. Therefore, future research should explore the use of various temporal features to improve the accuracy of prediction models for public emergencies.

 
 
 

#### 3.1.3. Participation Time

 
 The first arrival time t 1 t_{1} is another crucial temporal feature to consider in forecasting public emergencies. Moreover, various more sophisticated temporal features have been proposed in the literature, such as mean arrival time 1 / M ∑ M j = 1 t j 1/M{\textstyle\sum_{M}^{j=1}}t_{j} , mean reaction time 1 / M ∑ M j = 1 ( t j − t j − 1 ) 1/M{\textstyle\sum_{M}^{j=1}}(t_{j}-t_{j-1}) , change rate, dormant period, and peek fraction. For instance, research has shown that human reaction time typically follows a log-normal distribution, such as people’s reactions to calls, emails, and social networks ( Barabasi, 2005 ) . These temporal features provide useful insights for predicting the spread and impact of public emergencies.

 
 
 

#### 3.1.4. Evolving Trends

 
 Analyzing the changing trends of public emergencies has been shown to provide valuable insights for forecasting their severity and impact. Previous studies ( Gürsun et al., 2011 ; Leskovec et al., 2009 ; Yang and Leskovec, 2011 ) have demonstrated that identifying the temporal patterns of these events helps predict their popularity and impact on society. These patterns are categorized into various types, such as steadily increasing or rapidly fluctuating, depending on the clustering algorithm.

 
 
 
 

### 3.2. Structural Features

 
 The spread of public emergencies, such as pandemics, has garnered significant attention from scholars across various fields. In particular, the structure of cascades, also known as information diffusion, has been the subject of extensive research. It explains how information and contagions spread through populations ( Bao et al., 2013 ; Cheng et al., 2014 ; Galuba et al., 2010 ; Gao et al., 2014a ; Zaman et al., 2014 ; Zhang et al., 2016 ) . Scholars have approached the modeling of cascades in different ways, which are categorized into three types. The first type is Cascade Graph ( Pillai et al., 2016 ; Asim et al., 2016 ; Caigny et al., 2020 ; Zhao et al., 2016 ) , which involves the analysis of cascade graphs to understand how information spreads among participants in a network. The second type is Global Graph ( Chan and Franklin, 2011 ; Decroos et al., 2017 ; Deep et al., 2020 ; Ding et al., 2018 ; ElRefai et al., 2022 ; Allison, 2018 ; Gallego-Castillo et al., 2015 ; Gao and Zhao, 2018 ; Laxman et al., 2008 ; Minor and Cook, 2017 ; Piraján et al., 2019 ; Povinelli and Feng, 2003 ; Vahedian et al., 2017 ) , which considers both participants and non-participants in the network, considering the external factors that influence the spread of the emergency. Finally, the third type is r-reachable modeling ( Deep et al., 2020 ; Lin et al., 2018 ) , which seeks to strike a balance between the first two types, by expanding the cascade graph to include non-participants within the global graph as shown in Table 2 .

 
 

#### 3.2.1. Cascade Graph

 
 A cascade graph is constructed based on its participants and their interactions:

 
 
 Definition3.1 (Cascade Graph) Given an information item I i I_{i} and the corresponding cascade C i C_{i} , a cascade graph is defined as G c G_{c} = { V c , E c } =\{V_{c},E_{c}\} , where nodes V ​ c Vc = { u 0 , u 1 , … , u N } =\{u_{0},u_{1},...,u_{N}\} are participants of cascade C i C_{i} , and matrix E c E_{c} ⊂ V c ∗ V c \subset V_{c}*V_{c} contains a set of edges representing immediate relationships between V c V_{c} in a cascade.

 
 
 During public emergencies, the spread of information is crucial to inform and mobilize individuals to take necessary actions. The study of information diffusion through cascades has been widely researched, with a cascade graph being a fundamental concept in the field ( Caigny et al., 2020 ; Zhao et al., 2016 ) . A cascade graph, denoted as G c = V c , E c G_{c}={V_{c},E_{c}} , consists of a set of nodes, V c V_{c} , representing participants of the cascade, and a set of edges, E c E_{c} , representing the immediate relationships between the participants as shown in Table 3 . Cascade graphs play a vital role in characterizing the process of information diffusion of a particular item, such as the spreading directions and graph topology. For instance, one study examined the correlation between cascade popularity and two structural features, namely edge density, and depth, among early participants in microblogging networks ( Asim et al., 2016 ) . However, it is essential to note that the topological structure of cascade graphs varies significantly, even when the number of nodes is in the same ( Zhao et al., 2016 ) . The depth, structural virality, and other structural measurements, such as node degree or PageRank, may not accurately predict whether an information item would be popular. As cascades grow over time, the initial structural features may become less important in determining their popularity ( Pillai et al., 2016 ) . Therefore, further research is needed to explore the complex mechanisms of information diffusion during public emergencies and develop effective information dissemination strategies.

 
 
 

#### 3.2.2. Global Graph

 
 Table 3. Structural Features of Global Graphs 
 
 
 
 
 Structural features 
 | 
 
 
 Definition 
 | 
 
 
 Examples 
 | 
 
 
 References 
 | 

 
 
 
 Follower/Followee Graph 
 | 
 
 
 A graph where nodes represent users and edges represent the relationships 
 | 
 
 
 Twitter, Instagram, Facebook 
 | 
 
 
 ( Gallego-Castillo et al., 2015 ; Vahedian et al., 2017 ) 
 | 

 
 
 
 Hidden Friend Graph 
 | 
 
 
 A graph representing less obvious or non-publicly visible relationships 
 | 
 
 
 Twitter, Facebook 
 | 
 
 
 ( Chan and Franklin, 2011 ; Decroos et al., 2017 ; Ding et al., 2018 ; Minor and Cook, 2017 ; Piraján et al., 2019 ) 
 | 

 
 
 
 Co-Participation Graph 
 | 
 
 
 A graph that captures the relationships between users based on their participation 
 | 
 
 
 Digg, Reddit 
 | 
 
 
 ( Gao and Zhao, 2018 ; Gao et al., 2019b ; Granroth-Wilding and Clark, 2016 ) 
 | 

 
 
 
 Interaction Global Graph 
 | 
 
 
 A graph that captures interactions between users 
 | 
 
 
 Twitter, YouTube, Instagram 
 | 
 
 
 ( Decroos et al., 2017 ; Deep et al., 2020 ; Allison, 2018 ; Laxman et al., 2008 ; Povinelli and Feng, 2003 ) 
 | 

 
 
 Understanding the spread of information across different types of networks is crucial for effective crisis communication as shown in Table 3 . While cascade graphs provide insight into the local spread of information, exploring global graphs is also important ( Chan and Franklin, 2011 ; Decroos et al., 2017 ; Ding et al., 2018 ; Minor and Cook, 2017 ; Piraján et al., 2019 ) . These interaction global graphs are useful in various prediction tasks, particularly when the explicit social graph is not available, and historical behaviors serve as a suitable representation of the actual diffusion of information ( Vahedian et al., 2017 ) . Therefore, understanding the characteristics of different types of global graphs can aid in designing effective crisis communication strategies during public emergencies.

 
 
 Definition 3.2 (Global Graph) A global graph G g = ( V g , E g ) G_{g}=(V_{g},E_{g}) is a fundamental representation of the relationships between nodes, where V g V_{g} is a set of nodes, and E g ⊂ V g ∗ V g E_{g}\subset V_{g}*V_{g} is a set of edges that connect nodes based on their relationships. The global graph is further defined based on additional characteristics such as edge direction, edge weight, node/edge attributes, and node/edge features.

 
 
 By providing a macro perspective of the relationships between nodes, a global graph offers a valuable tool to analyze the spread of information to individuals and communities. In contrast to the cascade graph, which shows the local spread patterns for information cascade, the global graph describes the relationships between users and potential routes for diffusion ( Wang et al., 2020e ) . In social networking platforms, the discovery and dissemination of information items primarily occur through the users’ social networks. This highlights the importance of understanding the relationships between users in the global graph to effectively analyze and mitigate public emergencies such as the spread of misinformation, epidemics, and natural disasters. ( Sakaki et al., 2010 ) 

 
 
 

#### 3.2.3. r-reachable Graph

 
 Drawing upon the concepts of cascade and global graphs, a sub-graph extracted from the global graph, termed an r-reachable graph, is defined as follows:

 
 
 Definition 3.3 (r-reachable Graph) Consider a global graph G g G_{g} and its cascade sub-graph G c G_{c} . An r-reachable graph of G c G_{c} , denoted by G c r = V c r , E c r G_{c}^{r}={V_{c}^{r},E_{c}^{r}} , is a graph comprising the nodes in V c V_{c} and those in V g V_{g} that are within r r -hops of nodes in V c V_{c} . For instance, if r r = 1, then V c r V_{c}^{r} contains all nodes in V c V_{c} and their immediate neighbors.

 
 
 The r-reachable graph has been employed in several studies to aid in predicting the popularity of cascades in social media ( Lin et al., 2018 ) . A global graph G g G_{g} and several 1-reachable graphs G c 1 G_{c}^{1} were constructed from retweet cascades in Weibo, and structural features were extracted from G g G_{g} and G c 1 G_{c}^{1} to forecast tweet popularity ( Deep et al., 2020 ) . A comparative analysis of several content and structural features revealed that the size of the cascade graph and the 1-reachable graph are the two most predictive of 53 features for predicting the popularity of Twitter hashtags ( Lin et al., 2018 ) . Nevertheless, constructing and computing an r-reachable graph are computationally intensive; for instance, the 2-reachable graph of a cascade graph with dozens of nodes may contain tens of thousands of nodes ( Wang et al., 2019a ) .

 
 
 

#### 3.2.4. Summary

 
 Understanding the early structure and patterns of information diffusion is crucial for effective crisis management and response ( Chan and Franklin, 2011 ) . As the spread of misinformation and rumors exacerbate the impact of a crisis, identifying influential users and predicting the popularity of information items are essential tasks ( Do et al., 2015 ; Kruengkrai et al., 2017 ; Kupilik and Witmer, 2018 ) . Previous studies have shown that the structure of cascade graphs is not a reliable predictor of final item popularity, and alternative approaches such as identifying influential users must be considered. Furthermore, the types of items and their contents also affect information diffusion behavior during crises ( Santos et al., 2014 ) . Therefore, analyzing the structural patterns of information diffusion and identifying influential users in the context of public emergencies provide valuable insights for crisis management and response ( Laxman et al., 2008 ; Pillai et al., 2016 ) .

 
 
 
 

### 3.3. User/Item Features

 
 The unprecedented occurrence of public emergencies has heightened the need for accurate and timely predictions of their impact on society. However, obtaining early observations and monitoring temporal and structural features is impractical in the context of rapidly unfolding events. As such, some works try to examine the features inherent to users and information items, which possess distinct characteristics and inherent appeal that render them valuable in forecasting popularity before dissemination.

 
 

#### 3.3.1. User Features

 
 During public emergencies, such as natural disasters or pandemics, information dissemination becomes crucial in managing and mitigating the situation. User behaviors play a critical role in this process, as they determine the speed and reach of information spread. However, it is important to note that large cascades of information are not produced by influential users such as celebrities and news organizations but also originate from normal users. Therefore, it is essential to study and analyze large cascades produced by many types of users during public emergencies. Various other features have been extensively explored and studied for analyzing and predicting the popularity of information items during public emergencies. These features include user profiles, historical behaviors, user interests, collectivity, similarity, activity/passivity, discoveries, affinities, and responsiveness ( Fülöp et al., 2012 ; Mallouhy et al., 2019 ; Qi et al., 2018 ; Reyes et al., 2013 ; Shen et al., [n. d.] ; Yi et al., 2019 ; Zafarani et al., 2014 ) . Understanding and utilizing these features predict and manage the spread of information during public emergencies, ultimately leading to more effective communication and response.

 
 
 

#### 3.3.2. Item Features

 
 Understanding the impact of project characteristics on information dissemination has a significant impact on the dissemination of key messages as shown in Figure 3 . Previous literature explored various item characteristics and their impact on information dissemination. For example, one study analyzed how user interfaces on social media platforms affect item visibility ( Kleijnen and van Beers, 2020 ) . Furthermore, using entropy computed across information categories and topics provides insight into the diversity of information and its impact on dissemination ( Gao and Zhao, 2018 ) . Using popularity variability to analyze how the popularity of different types of items changes over time helps identify patterns and trends in information dissemination during public emergencies ( Wang et al., 2015 ) .

 
 
 Figure 3. From Emergency Time-line to Transactions 
 
 
 

#### 3.3.3. Summary

 
 During public emergencies, understanding the factors that contribute to information diffusion is critical. User and item characteristics play a significant role, with some features being more easily understood than others. However, factors like user influence, preferences, and similarities require more sophisticated algorithms and calculations. Advanced techniques enable better predictions and are particularly useful in such situations where timely and accurate information dissemination is crucial ( Jurgens, 2021 ; Bao et al., 2019 ; Chan and Lam, 2005 ) .

 
 
 
 

### 3.4. Content Features

 
 From the perspective of predicting public emergencies, content is widely recognized as an essential driving force and one of the key factors that contribute to the success of any emergency response effort. For instance, breaking news, rumors, and fake news, as well as hot spots and controversial or peculiar topics, disinformation, and misinformation, tend to attract considerably more attention than regular content.

 
 

#### 3.4.1. Text Content

 
 From the perspective of predicting public emergencies, text feature is a crucial component of existing prediction models. Textual information is pervasive in articles, microblogs, image/audio/video captions/descriptions, and even retrieved from multimedia sources. Researchers have employed various language models such as Term Frequency-Inverse Document Frequency (TF-IDF) and Latent Dirichlet Allocation (LDA) ( Liu et al., 2018 ) , along with typical Machine Learning models like naive Bayes, SVM, and linear regression to predict item popularity. TF-IDF and LDA are used to learn the topic distributions of tweets ( Fülöp et al., 2012 ) . TF-IDF is utilized to estimate the importance of keywords in user tweets and to calculate the mutual correlation between a user’s historical content and a specific item to measure their likelihood of adopting that item as shown in Figure 4(a) ( Yi et al., 2019 ) . Authors analyze various semantic and statistical content features of Digg comments to identify the characteristics of content that people prefer to retweet ( Gulmezoglu et al., 2019 ) .

 
 
 
 
 
 (a) Text Content 
 
 
 (b) Image Content 
 
 Figure 4. Content Features 
 
 
 

#### 3.4.2. Image Features

 
 The retrieval and analysis of image features require techniques from computer vision learning, distinct from text-based methods. Basic features of an image, such as size, date, orientation, dominant color, resolution, location, caption, and tag . For example, during the COVID-19 pandemic, computer vision techniques were utilized to analyze images from social media platforms and identify violations of social distancing protocols, such as large gatherings or people not wearing masks ( Li et al., 2007 ) . Lu et al. ( Lu et al., 2017 ) provide an analysis of these attributes, while Michael et al. ( Hagenau et al., 2012 ) study the correlations between Flickr image features and their normalized popularity as shown in Figure 4(b) . Content features of images, categorized as simple human-interpretable, low-, and high-level image features , contribute significantly to improving prediction performance. During natural disasters like hurricanes or earthquakes, computer vision techniques are utilized to analyze satellite images and identify areas that have been affected and need assistance. Li et al. ( Li et al., 2007 ) extract texture and color features and use VGG19 ( Ristea et al., 2020 ) to extract deep features. During a pandemic, computer vision techniques are utilized to analyze chest X-rays and identify patterns that could indicate COVID-19 infection.

 
 
 

#### 3.4.3. Summary

 
 The challenge of predicting item features based solely on content features is still prevalent, despite previous studies exploring the linguistic and visual characteristics of items. This difficulty arises because even items with the same content have varying features, making it challenging to distinguish the effects of descriptive factors from intrinsic content. This unpredictability is particularly relevant in predicting the spread of misinformation during a public health crisis.

 
 
 
 
 

## 4. Deduction of Public Emergencies

 
 Predicting the joint information of public emergencies is a crucial task in disaster management. Research in this area are categorized into six types: (1) Joint Time and Semantic Prediction; (2) Joint Time and Location Prediction; (3) Joint Time, Location, and Semantic Prediction; (4) Vulnerability Prediction; (5) Association-based Impact Prediction; (6) Causality-based Prediction.

 
 

### 4.1. Time and Semantics

 
 In the prediction of public emergencies, various methods have been developed to jointly predict the joint time and semantics. Existing work in this area is broadly categorized into the following three types: Temporal Association Rule; Spatial-Temporal Methods; Time Series Forecasting-based Methods. 

 
 

#### 4.1.1. Temporal Association Rule

 
 Temporal association rules are used to embed additional temporal information into the occurrence of events and to redefine the meaning of co-occurrence and association with temporal constraints as shown in Table 4 . In that case, they define the Left-Hand Side (LHS) of the association rule as a tuple ( E L , τ ) (E_{L},\tau) , where τ \tau represents the time window before the target flood emergency occurrence in the Right-Hand Side (RHS) predefined by the user. The events that occur within this time window before the flood emergency satisfy the LHS. However, defining the time window beforehand is challenging and may not suit different target events. To overcome this, researchers have proposed a way to automatically identify information on a continuous time interval from the data ( Shen et al., [n. d.] ) . Here, the transaction comprises items and continuous time duration information as shown in figure 6 . LHS is a set of items (e.g., previous emergencies), and RHS is a tuple ( E R , [ t 1 , t 2 ] ) (E_{R},[t_{1},t_{2}]) consisting of a future emergency’s semantic representation and its time interval of occurrence. To learn the time interval in RHS, two different methods have been proposed: confidence-interval-based and minimal temporal region selection ( Shen et al., [n. d.] ) .

 
 
 Table 4. Deduction of Public Emergencies 
 
 
 
 
 Category 
 | 
 
 
 Direction 
 | 
 
 
 Method 
 | 
 
 
 References 
 | 

 
 
 
 Risk Occurrence 
 | 
 
 
 Binary Classification 
 | 
 
 
 Simple Threshold-based 
 | 
 
 
 ( Wang et al., 2013 ; Wang and Zhang, 2017 ; Wang and Gerber, 2015 ; Wang et al., 2020b ) 
 | 

 
 | 
 | 
 
 
 Logistic regression 
 | 
 
 
 ( Wang et al., 2019b ; Yan et al., 2018 ; Wang and Ding, 2015 ) 
 | 

 
 | 
 | 
 
 
 Support Vector Machines 
 | 
 
 
 ( Gao et al., 2019b ; Ghil et al., 2011 ; Gao and Zhao, 2018 ) 
 | 

 
 | 
 | 
 
 
 Neural Networks 
 | 
 
 
 ( Kunneman et al., 2020 ; Kulldorff, 1997 ; Kupilik and Witmer, 2018 ) 
 | 

 
 | 
 | 
 
 
 Decision trees 
 | 
 
 
 ( Coglianese and Nash, 2016 ; Compton et al., 2014 ; ElRefai et al., 2022 ) 
 | 

 
 | 
 
 
 Anomaly detection 
 | 
 
 
 One-classification, hypotheses testing 
 | 
 
 
 ( Gallego-Castillo et al., 2015 ; Oki et al., 2018 ) 
 | 

 
 | 
 
 
 Regression 
 | 
 
 
 Auto-regression, linear regression, ordinal regression 
 | 
 
 
 ( Decroos et al., 2017 ; Doswell et al., 1993 ; Srinivasa et al., 2008 ) 
 | 

 
 
 
 Discrete-time 
 | 
 
 
 Time windows 
 | 
 
 
 Auto-regression, ARIMA, ordinal regression or classification 
 | 
 
 
 ( Li et al., 2016 ; Li et al., 2018c ; Petropoulos and Makridakis, 2020 ) 
 | 

 
 | 
 
 
 Time scales 
 | 
 
 
 Regression or classification 
 | 
 
 
 ( Rebane et al., 2019 ; Reid et al., 2018 ; Reyes et al., 2013 ) 
 | 

 
 | 
 
 
 Time series 
 | 
 
 
 Autoregressive, burstiness detection, change detection 
 | 
 
 
 ( Barnes et al., 2016 ; Bialonski et al., 2015 ; Han et al., 2012 ) 
 | 

 
 | 
 | 
 
 
 Learning event characterization 
 | 
 
 
 ( Jurgens, 2021 ; Jin et al., 2013 ; Kang et al., 2017 ; Kattan et al., 2015 ) 
 | 

 
 
 
 Rule-based 
 | 
 
 
 Associative learning 
 | 
 
 
 Frequent set mining 
 | 
 
 
 ( Han et al., 2012 ; Letham et al., 2015 ; Vilalta and Ma, [n. d.] ; Zhou et al., 2015 ) 
 | 

 
 
 
 Cross-chain 
 | 
 
 
 Generalizing event graphs 
 | 
 
 
 Graph representation 
 | 
 
 
 ( Antunes et al., 2003 ; Su and Jiang, 2020 ) 
 | 

 
 
 
 Semantic causation 
 | 
 
 
 Event representation 
 | 
 
 
 RDF-based 
 | 
 
 
 ( Caigny et al., 2018 ; Chan and Franklin, 2011 ; Khoo et al., 2000 ; Kim, 1993 ; Zhao et al., 2017 ) 
 | 

 
 | 
 
 
 Event inference 
 | 
 
 
 probability estimate 
 | 
 
 
 ( Acharya et al., 2017 ; Nguyen et al., 2017a ; Radinsky et al., 2012 ) 
 | 

 
 | 
 
 
 Future event inference 
 | 
 
 
 Circular binary search tree 
 | 
 
 
 ( Choi et al., 2018 ; Lei et al., 2019 ; Radinsky et al., 2012 ; Zhao et al., 2017 ) 
 | 

 
 
 
 Semantic model 
 | 
 
 
 Feature-based 
 | 
 
 
 Aggregation-based 
 | 
 
 
 ( Tama and Comuzzi, 2019 ; Taylor, 2017 ; van Noord et al., 2017 ; Vahedian et al., 2017 ) 
 | 

 
 | 
 | 
 
 
 Compositional-based 
 | 
 
 
 ( Fronza et al., 2013 ; Granroth-Wilding and Clark, 2016 ) 
 | 

 
 | 
 | 
 
 
 Markov models 
 | 
 
 
 ( Alevizos et al., 2017 ; Alevizos et al., 2018 ; Laxman et al., 2008 ; Pillai et al., 2016 ; Yang et al., 2014 ) 
 | 

 
 | 
 
 
 Prototype-based 
 | 
 
 
 Clustering-based 
 | 
 
 
 ( Adhikari et al., 2019 ; Abuella and Chowdhury, 2019 ; Boni and Gerber, 2016 ) 
 | 

 
 | 
 
 
 Attribute-based 
 | 
 
 
 Feature extraction-based 
 | 
 
 
 ( Casagrande et al., 2018 ; Heaton, 2017 ; Lin et al., 2019 ; Li et al., 2018c ) 
 | 

 
 | 
 
 
 Descriptive-based 
 | 
 
 
 Text-based 
 | 
 
 
 ( Hu, 2020 ; Hu et al., 2017 ; Lv et al., 2019 ; Su and Jiang, 2020 ; Xue et al., 2018 ; Yu et al., 2019 ) 
 | 

 
 
 

#### 4.1.2. Spatial-Temporal Methods

 
 Spatial-temporal methods are used to predict public emergencies by incorporating spatial and temporal information about emergencies. For instance, a spatio-temporal model is proposed to predict emergency incidence ( Calabrese and Elkink, 2016 ) . It considers the correlation between the spatial and temporal dimensions of emergencies and incorporates spatial factors such as population density and road network density. A multi-view learning-based spatio-temporal model is proposed, which learns the features of spatial and temporal domains separately and then fuses them for prediction ( Salfner and Malek, 2007 ) . Another approach is to leverage the deep learning model for prediction ( Srinivasa et al., 2008 ) . A spatiotemporal Graph Convolutional Network is proposed that learns the spatial and temporal correlations of emergencies using graph convolutional networks as shown in Figure 6 .

 
 
 
 
 
 Figure 5. Tensor Decomposition and Forecasting for Complex Time-stamped events 
 
 
 Figure 6. Generic framework for hierarchical RNN-based information cascade prediction under public emergencies 
 
 
 
 

#### 4.1.3. Time Series Forecasting-based Methods

 
 Methods for forecasting public emergencies based on time series are divided into two categories:

 
 
 Direct methods typically approach the problem of predicting emergency types as a multivariate time series forecasting problem, where each variable corresponds to an emergency type E i ​ ( i = 1 , … ) E_{i}(i=1,...) and the predicted emergency type at future time t t is calculated as s ^ t ^ = a ​ r ​ g ​ m ​ a ​ x ​ E i \hat{s}_{\hat{t}}=argmax{E_{i}} f ⁡ ( s t ^ = E i | X ) f(s_{\hat{t}}=E_{i}|X) . For instance, in ( Li et al., 2017 ) , a longitudinal support vector regressor is utilized to forecast multi-dimensional emergencies. They build n n Support Vector Regressors, each of which corresponds to a specific attribute to achieve the goal of predicting the next time point’s attribute value. To predict multiple emergency types, Weiss and Page ( Weiss and Page, 2013 ) use multiple-point process models. To improve the accuracy of their predictions, Biloš et al. ( Ma and Leung, 2019 ) employ RNN to learn the historical representation of emergencies, and then input the results into a Gaussian process model to predict future emergency types. To better capture the dynamics across multiple variables in the time series, Brandt et al. ( Brandt et al., 2011 ) extend this to Bayesian vector autoregression.

 
 
 Indirect methods focus on learning a mapping from observed emergency types to low-dimensional latent-topic space using tensor decomposition-based techniques. Regarding information cascade prediction under public emergencies, Matsubara et al. ( Matsubara et al., 2012a ) propose a 3-way analysis of the original observed emergency tensor Y 0 ∈ R D o ​ D a ​ D e Y_{0}\in R^{D_{o}D_{a}D_{e}} , which consists of three factors: actors, objects, and time. They decompose this tensor into latent variables via three corresponding low-rank matrices P o ∈ R D k ​ D o P_{o}\in R^{D_{k}D_{o}} , P a ∈ R D k ​ D a P_{a}\in R_{D_{k}D_{a}} , and P e ∈ R D k ∗ D e P_{e}\in R_{D_{k}*D_{e}} , where D k D_{k} is the number of latent topics. Through multivariate time series forecasting, the time matrices P e P_{e} are predicted into the future to estimate future emergency tensors by recovering a "future emergency tensor" Y ^ \hat{Y} through the multiplication of the predicted time matrix P e P_{e} with the known actor matrix P a P_{a} and object matrix P o P_{o} . This approach provides insights into the different types of emergencies based on latent topics.

 
 
 
 

### 4.2. Time and Location

 
 The prediction of the location and time of future public emergencies is a critical area of research for emergency management and disaster response. One relevant category of methods for such predictions is based on the joint consideration of time and location. These methods are classified into two subtypes based on their approach as shown in Figure 7 .

 
 
 Figure 7. The Type Of Location-based Prediction 
 
 

#### 4.2.1. Grid-based

 
 Grid-based methods are useful in predicting the location and time of future public emergencies. In recent years, several techniques have been proposed to capture spatial and temporal information for information cascade prediction under public emergencies as shown in Table 5 .

 
 
 Table 5. Grid-based Location Prediction 
 
 
 
 
 Category 
 | 
 
 
 Description 
 | 
 
 
 References 
 | 

 
 
 
 Spatial Clustering 
 | 
 
 
 Grouping contiguous regions that collectively exhibit significant patterns. 
 | 
 
 
 ( Jiang, 2019 ; Wang and Ding, 2015 ; Xiong et al., 2019 ) 
 | 

 
 
 
 Spatial Interpolation 
 | 
 
 
 Estimating the probability of event occurrence at locations without historical data, resulting in spatial smoothness. 
 | 
 
 
 ( Boni and Gerber, 2016 ; Hao et al., 2019 ; Jiang, 2019 ; Kleijnen and van Beers, 2020 ; Ristea et al., 2020 ) 
 | 

 
 
 
 Spatial Convolution 
 | 
 
 
 Learning and representing spatial patterns using Convolutional Neural Networks (CNNs). 
 | 
 
 
 ( Cloke and Pappenberger, 2009 ; Heaton, 2017 ; Bao et al., 2019 ; Mukhina et al., 2019 ; Piraján et al., 2019 ; Wang et al., 2019b ) 
 | 

 
 
 
 Trajectory Destination 
 | 
 
 
 Predicting population-based events by interpreting the collective behaviors of individuals. 
 | 
 
 
 ( Wang and Gerber, 2015 ; Vahedian et al., 2017 ; Kulldorff, 1997 ; Xiong et al., 2019 ) 
 | 

 
 
 A straightforward approach to incorporating spatial information is to include location data as an input feature and use it in predictive models, such as Linear Regression ( Zhao and Tang, 2017 ) , LSTM ( Ren et al., 2018 ) , and Gaussian Processes ( Kupilik and Witmer, 2018 ) . Zhao et al. ( Zhao and Tang, 2017 ) utilized the spatiotemporal dependency to regularize their model parameters during training. However, most of the methods in this domain aim to jointly consider the spatial and temporal dependencies for predictions ( Di et al., 2019 ) . To achieve this, the multi-attributed spatial information for each time point is organized as a series of multi-channel images that are encoded using convolution-based operations ( Fang et al., 2020 ) .

 
 
 The multi-attributed spatial information is represented as a spatiotemporal grid, where each cell in the grid represents a specific location and time point. Let X i , j , t X_{i,j,t} denote the value of the i i -th attribute at the ( i , j ) (i,j) -th location and t t -th time point. The spatiotemporal grid is represented as:

 

 
 (1) | 
 | 
 X = [ X i , j , t ] i = 1 , … , n ; j = 1 , … , m ; t = 1 , … , T X=[X_{i,j,t}]_{i=1,\dots,n;j=1,\dots,m;t=1,\dots,T} | 
 | 
 

 where n n and m m represent the number of locations in the grid, and T T represents the number of time points.

 
 
 To encode the spatiotemporal grid using convolution-based operations, it first converts it into a series of multi-channel images, where each channel corresponds to a different attribute. Let X ( k ) X^{(k)} denote the k k -th channel of the image, which is a 2D matrix of size n × m n\times m :

 

 
 (2) | 
 | 
 X ( k ) = [ X i , j , t ( k ) ] i = 1 , … , n ; j = 1 , … , m X^{(k)}=[X_{i,j,t}^{(k)}]_{i=1,\dots,n;j=1,\dots,m} | 
 | 
 

 
 
 The 2D convolution operation is then applied to each channel of the image to learn spatial features. The output of the convolution operation is a 2D feature map, which is then passed through a nonlinear activation function. Let W ( k ) W^{(k)} denote the weight matrix for the k k -th channel, and b ( k ) b^{(k)} denote the bias vector. The output of the convolution operation is expressed as:

 

 
 (3) | 
 | 
 H i , j , t ( k ) = f ⁡ ( ∑ p = 1 P ∑ q = 1 Q W p , q ( k ) ​ X i + p − 1 , j + q − 1 , t + b ( k ) ) H^{(k)}_{i,j,t}=f\left(\sum_{p=1}^{P}\sum_{q=1}^{Q}W^{(k)}_{p,q}X_{i+p-1,j+q-1,t}+b^{(k)}\right) | 
 | 
 

 where P P and Q Q represent the size of the convolution kernel, and f f is the activation function.

 
 
 The output of the convolution operation is then passed through a pooling layer to reduce the spatial dimensionality of the feature map. The pooling operation computes a summary statistic over a local neighborhood of the feature map. The most common pooling operations are max pooling and average pooling. Let H i , j , t ( k ) H^{(k)}_{i,j,t} denote the output of the pooling operation for the k k -th channel. The output of the pooling layer is expressed as:

 

 
 (4) | 
 | 
 Y i , j , t ( k ) = pool ​ ( { H p , q , t ( k ) } p = i , … , i + K − 1 ; q = j , … , j + K − 1 ) Y^{(k)}_{i,j,t}=\text{pool}\left(\{H^{(k)}_{p,q,t}\}_{p=i,\dots,i+K-1;q=j,\dots,j+K-1}\right) | 
 | 
 

 where K K represents the size of the pooling window, and pool is the pooling function.

 
 
 The outputs of the pooling layer for channels are concatenated to form a spatiotemporal feature vector. Let Y t Y_{t} denote the spatiotemporal feature vector at time t t . The spatiotemporal feature vector is expressed as:

 

 
 (5) | 
 | 
 Y t = [ Y 1 , 1 , t ( 1 ) , … , Y n , m , t ( 1 ) , Y 1 , 1 , t ( 2 ) , … , Y n , m , t ( 2 ) , … , Y 1 , 1 , t ( K ) , … , Y n , m , t ( K ) ] Y_{t}=[Y^{(1)}_{1,1,t},\dots,Y^{(1)}_{n,m,t},Y^{(2)}_{1,1,t},\dots,Y^{(2)}_{n,m,t},\dots,Y^{(K)}_{1,1,t},\dots,Y^{(K)}_{n,m,t}] | 
 | 
 

 
 
 

#### 4.2.2. Point-based

 
 Spatio-temporal point process modeling is a valuable technique for predicting the time and location of public emergencies. This method models the rate of event occurrence in both space and time, providing a comprehensive understanding of the spatiotemporal distribution of events. The technique is defined as:

 

 
 (6) | 
 | 
 λ ⁡ ( t , l | X ) = lim | d ​ t | → 0 , | d ​ l | → 0 E ⁡ [ N ⁡ ( d ​ t ∗ d ​ l ) | X ] ( | d ​ t | ​ | d ​ l | ) \lambda(t,l|X)=\lim_{|dt|\rightarrow 0,|dl|\rightarrow 0}\frac{E[N(dt*dl)|X]}{(|dt||dl|)} | 
 | 
 

 Several models have been proposed to instantiate this framework, which includes different facets of input data such as location, time, and other semantic features. Liu and Brown et al. ( Liu and Brown, 2004 ) assumed conditional independence among spatial and temporal factors and decomposed the rate of event occurrence as follows:

 

 
 (7) | 
 | 
 λ ( t , l | X ) = λ ( t , l | L , T , F ) = λ 1 ( l | L , T , F , t ) ∗ λ 2 ( t | T ) \lambda(t,l|X)=\lambda(t,l|L,T,F)=\lambda_{1}(l|L,T,F,t)*\lambda_{2}(t|T) | 
 | 
 

 In this equation, λ 1 ​ ( x ) \lambda_{1}(x) is modeled using the Markov spatial point process, while λ 2 ​ ( x ) \lambda_{2}(x) is characterized using temporal autoregressive models. To handle situations where explicit assumptions for model distributions are difficult, several methods have been proposed to involve deep architecture during the point process. Recently, Okawa et al. ( Okawa et al., 2019 ) proposed the following:

 

 
 (8) | 
 | 
 λ ⁡ ( t , l ∣ X ) = ∫ g θ ​ ( t ′ , l ′ , ℱ ⁡ ( t ′ , l ′ ) ) ⋅ 𝒦 ⁡ ( ( t , l ) , ( t ′ , l ′ ) ) ​ d ​ t ′ ​ d ​ l ′ \lambda(t,l\mid X)=\int g_{\theta}\left(t^{\prime},l^{\prime},\mathcal{F}\left(t^{\prime},l^{\prime}\right)\right)\cdot\mathcal{K}\left((t,l),\left(t^{\prime},l^{\prime}\right)\right)\mathrm{d}t^{\prime}\mathrm{d}l^{\prime} | 
 | 
 

 Here, K ⁡ ( x , y ) K(x,y) is a kernel function, such as a Gaussian kernel ( Bishop and Nasrabadi, 2006 ) , which measures the similarity in time and location dimensions. F ⁡ ( t , l ) ⊆ F F(t,l)\subseteq F denotes the feature values for the data at location l ′ l^{{}^{\prime}} and time t ′ t^{{}^{\prime}} . g θ ​ ( x ) g_{\theta}(x) is a deep neural network that parameterized by θ \theta and returns a nonnegative scalar. The model selection of g θ ​ ( x ) g_{\theta}(x) depends on the data types.

 
 
 
 

### 4.3. Time, Location, and Semantics

 
 Table 6. Time, Location, and Semantics 
 
 
 
 
 Category 
 | 
 
 
 Directions 
 | 
 
 
 Methods 
 | 
 
 
 References 
 | 

 
 
 
 System-based 
 | 
 
 
 Event prediction 
 | 
 
 
 Model-fusion system 
 | 
 
 
 ( Ramakrishnan et al., 2014 ; Hoegh et al., 2015 ; Kang et al., 2017 ) 
 | 

 
 | 
 
 
 Crowd-sourced system 
 | 
 
 
 Recommender system 
 | 
 
 
 ( Rostami et al., 2018 ; Reyes et al., 2013 ; Rouet-Leduc et al., 2017 ) 
 | 

 
 | 
 
 
 Future event detection 
 | 
 
 
 Prediction market system 
 | 
 
 
 ( Li et al., 2016 ; Letham et al., 2013 ; Li et al., 2017 ) 
 | 

 
 
 
 Model-based 
 | 
 
 
 Planned event detection 
 | 
 
 
 Mathematical models 
 | 
 
 
 ( Kruengkrai et al., 2017 ; Mirtaheri et al., 2019 ; Bhattacharjya et al., 2020 ) 
 | 

 
 | 
 | 
 
 
 Content filtering 
 | 
 
 
 ( Hürriyetoǧlu et al., 2017 ; Nakajima et al., 2019 ) 
 | 

 
 | 
 | 
 
 
 Time expression identification 
 | 
 
 
 ( Damaschke et al., 2017 ; Hürriyetoǧlu et al., 2017 ) 
 | 

 
 | 
 | 
 
 
 Future reference sentence extraction 
 | 
 
 
 ( Kunneman et al., 2020 ; Nakajima et al., 2019 ) 
 | 

 
 | 
 | 
 
 
 Location identification 
 | 
 
 
 ( Becker et al., 2012 ; ElRefai et al., 2022 ; Jurgens, 2021 ; Muthiah et al., 2016 ) 
 | 

 
 
 
 Tensor-based 
 | 
 
 
 Tensor extrapolation 
 | 
 
 
 Tensor decomposition 
 | 
 
 
 ( Mirtaheri et al., 2019 ; Zhou et al., 2019 ; Zhou et al., 2015 ) 
 | 

 
 | 
 | 
 
 
 Tensor completion 
 | 
 
 
 ( Zhou et al., 2019 ; Zhao and Tang, 2017 ) 
 | 

 
 
 In this section, we discuss various techniques that are utilized for predicting the time, location, and Semantics of public emergencies. These methods are broadly classified into two categories: System-based approaches rely on the development and deployment of complex systems that are designed to gather and analyze data from various sources in real time. On the other hand, Future Event Detection Methods approaches involve the development of mathematical models that use historical data to predict future emergencies. Both approaches have their own strengths and limitations, and the choice of strategy depends on the specific needs and constraints of the emergency management system.

 
 

#### 4.3.1. System-based Approaches

 
 Public emergencies are predicted using system-based or model-based strategies. In this section, we discuss system-based approaches. One such approach is the model-fusion system, which integrates various techniques for predicting the time, location, and semantics of public emergencies into a unified emergencies prediction system ( Wang et al., 2020d ) .

 
 
 Cascaded prediction of emergency events is crucial for timely and effective response. EMBERS ( Ramakrishnan et al., 2014 ) is an online warning system that predicts the time, location, and type of future events, as well as the population affected. To maximize precision, EMBERS prioritizes individual prediction models by suppressing their recall ( Wang et al., 2021b ) . The fusion of predictions from different models eventually results in high recall. Bayesian fusion-based strategies have been investigated ( Hoegh et al., 2015 ) , and similar strategies are used in systems like Carbon ( Kang et al., 2017 ) . Crowd-sourced systems are another approach that implements fusion strategies to generate predictions made by human predictors. For instance, Rostami et al. ( Rostami et al., 2018 ) proposed a recommender system that matches event-predicting tasks to human predictors with suitable skills to maximize the accuracy of their fused predictions. This approach addresses the heterogeneity and diversity of human predictors’ skill sets and background knowledge under limited human resources.

 
 
 

#### 4.3.2. Future Event Detection Methods

 
 Planned public emergencies are detected using methods that rely on Natural Language Processing (NLP) techniques and Linguistic Principles to analyze various media sources such as social media and news. These methods typically consist of four main steps.

 
 ∙ \bullet 
 
 Content filtering is employed to retain the texts that are relevant to the event of interest. Existing works utilize either supervised methods or unsupervised methods.

 

 ∙ \bullet 
 
 Time expression identification is used to identify future reference expressions and determine the time of the event. This step leverage existing tools such as the Rosetta text analyzer ( Damaschke et al., 2017 ) or propose dedicated strategies based on linguistic rules ( Hürriyetoǧlu et al., 2017 ) .

 

 ∙ \bullet 
 
 Future reference sentence extraction is the core of planned event detection and is implemented either by designing regular expression-based rules ( Nakajima et al., 2019 ) or by textual classification ( Kunneman et al., 2020 ) .

 

 ∙ \bullet 
 
 Location identification is essential to infer the event’s location accurately. The expression of locations is typically highly heterogeneous and noisy. Existing works have relied heavily on geocoding techniques to resolve the event location accurately. Various types of locations are considered, such as article locations, locations mentioned in the articles ( Becker et al., 2012 ) , and authors’ neighbors’ locations ( Jurgens, 2021 ) . Multiple locations have been selected using a geometric median ( ElRefai et al., 2022 ) or fused using logical rules such as probabilistic soft logic ( Muthiah et al., 2016 ) .

 

 
 
 
 

#### 4.3.3. Tensor-based Methods

 
 Public emergencies are often analyzed using tensor decomposition and extrapolation techniques. These methods involve formulating the data into a tensor form, which includes dimensions such as location, time, and semantics. Tensor decomposition is then applied to approximate the original tensor by using multiple low-rank matrices, each of which represents a mapping from latent topics to each dimension ( Wang et al., 2021a ) . Specifically, given a tensor 𝒯 \mathcal{T} with dimensions I I , J J , and K K , the low-rank tensor approximation is represented as:

 

 
 (9) | 
 | 
 𝒯 ≈ ∑ r = 1 R 𝐀 r ( 1 ) ∘ 𝐀 r ( 2 ) ∘ 𝐀 r ( 3 ) \mathcal{T}\approx\sum_{r=1}^{R}\mathbf{A}^{(1)}_{r}\circ\mathbf{A}^{(2)}_{r}\circ\mathbf{A}^{(3)}_{r} | 
 | 
 

 where 𝐀 r ( 1 ) ∈ ℝ I × R \mathbf{A}^{(1)}_{r}\in\mathbb{R}^{I\times R} , 𝐀 r ( 2 ) ∈ ℝ J × R \mathbf{A}^{(2)}_{r}\in\mathbb{R}^{J\times R} , and 𝐀 r ( 3 ) ∈ ℝ K × R \mathbf{A}^{(3)}_{r}\in\mathbb{R}^{K\times R} are low-rank matrices, ∘ \circ denotes the outer product, and R R is the rank of the tensor.

 
 
 After the tensor is decomposed, various strategies are employed to extrapolate the tensor toward future periods. For instance, Mirtaheri ( Mirtaheri et al., 2019 ) extrapolated the time dimension matrix and then multiplied it with the other dimensions’ matrices to recover the estimated extrapolated tensor into the future:

 

 
 (10) | 
 | 
 𝐀 r ( 3 ) ​ ( t f ) ≈ 𝐀 r ( 3 ) ​ ( t n ) + ( t f − t n ) ​ Δ ​ 𝐀 r ( 3 ) \mathbf{A}^{(3)}_{r}(t_{f})\approx\mathbf{A}^{(3)}_{r}(t_{n})+(t_{f}-t_{n})\Delta\mathbf{A}^{(3)}_{r} | 
 | 
 

 where 𝐀 r ( 3 ) ​ ( t f ) \mathbf{A}^{(3)}_{r}(t_{f}) and 𝐀 r ( 3 ) ​ ( t n ) \mathbf{A}^{(3)}_{r}(t_{n}) are the time dimension matrices at the future period t f t_{f} and the last observed period t n t_{n} , respectively, and Δ ​ 𝐀 r ( 3 ) \Delta\mathbf{A}^{(3)}_{r} is the first-order difference matrix of the time dimension matrix.

 
 
 On the other hand, Zhou et al. ( Zhou et al., 2019 ) adopted a different approach, where they added "empty values" for the entries corresponding to a future time in the original tensor. Then, they utilized tensor completion techniques to infer the missing values that correspond to future events. These techniques include low-rank tensor completion and tensor regression, which aim to recover the missing entries by exploiting the low-rank structure and the relationships between the dimensions of the tensor. By extrapolating the tensor towards future periods, decision-makers better understand the potential impact of such events and prepare appropriate response plans.

 
 
 
 

### 4.4. Vulnerability Prediction

 
 Vulnerability prediction in the context of public emergencies refers to the use of machine learning models to identify and assess potential risks and vulnerabilities associated with various hazards, such as earthquakes, floods, landslides, and extreme weather events.

 
 
 In public emergencies, innovative solutions are required, and machine learning (ML) models have emerged as a promising approach to addressing the challenges posed by these events.

 
 
 Prasad et al. ( Prasad et al., 2021 ) utilized an ML-based ensemble technique that combines multiple models to improve the accuracy of flood vulnerability mapping. The ensemble method is expressed mathematically as follows:

 

 
 (11) | 
 | 
 F ⁡ ( x ) = ∑ t = 1 T w t ​ f t ​ ( x ) F(x)=\sum_{t=1}^{T}w_{t}f_{t}(x) | 
 | 
 

 where x x represents the input data, F ⁡ ( x ) F(x) is the output of the ensemble model, f t ​ ( x ) f_{t}(x) is the output of the t t -th base model, and w t w_{t} is the weight assigned to the t t -th base model. The weights are typically computed based on the performance of the base models on the training data.

 
 
 Similarly, Nsengiyumva and Valentino ( Nsengiyumva and Valentino, 2020 ) employed the logistic model tree (LMT) to predict the vulnerability of landslide areas. The LMT is expressed mathematically as follows:

 

 
 (12) | 
 | 
 p ⁡ ( C k | x ) = N k ​ ( x ) N ⁡ ( x ) p(C_{k}|x)=\frac{N_{k}(x)}{N(x)} | 
 | 
 

 where p ⁡ ( C k | x ) p(C_{k}|x) is the probability of the k k -th class given the input data x x , N k ​ ( x ) N_{k}(x) is the number of training instances in class k k that satisfy the conditions specified by the tree, and N ⁡ ( x ) N(x) is the total number of training instances that satisfy the conditions specified by the tree.

 
 
 

### 4.5. Association-based Impact Prediction

 
 Association-based impact prediction in the context of public emergencies involves the use of association Rule-based methods to predict future events accurately. These methods rely on discovering associations between precursor and target events to predict the onset of various hazards, such as pandemics.

 
 
 The Apriori algorithm is a popular algorithm for frequent set mining, which generates frequent item sets by iteratively joining smaller itemsets and pruning infrequent ones. Mathematically, the support of an itemset I I is defined as:

 

 
 (13) | 
 | 
 S ​ u ​ p ​ p ​ ( I ) = f ​ r ​ e ​ q ​ ( I ) N Supp(I)=\frac{freq(I)}{N} | 
 | 
 

 where f ​ r ​ e ​ q ​ ( I ) freq(I) is the number of transactions that contain the itemset I I , and N N is the total number of transactions.

 
 
 Association rule mining involves finding rules of the form X → Y X\rightarrow Y , where X X is a set of items (the precursor event), and Y Y is a set of items (the target event). The association rules X → Y X\rightarrow Y are said to be valid if their support and confidence are above a given threshold. Pruning strategies are employed to remove invalid rules and retain the most accurate rules. The lift measure is another commonly used metric that measures the strength of association between the precursor event X X and the target event Y Y is defined as:

 

 
 (14) | 
 | 
 l ​ i ​ f ​ t ​ ( X → Y ) = s ​ u ​ p ​ p ​ ( X ∪ Y ) s ​ u ​ p ​ p ​ ( X ) ​ s ​ u ​ p ​ p ​ ( Y ) lift(X\rightarrow Y)=\frac{supp(X\cup Y)}{supp(X)supp(Y)} | 
 | 
 

 where s ​ u ​ p ​ p ​ ( X ∪ Y ) supp(X\cup Y) is the support of the itemset X ∪ Y X\cup Y . A lift value greater than 1 indicates a positive association between X X and Y Y . In conclusion, association-based methods effectively predict events during public emergencies by discovering associations between precursor and target events.

 
 
 

### 4.6. Causality-based Prediction

 
 Causality-based prediction involves inferring cause-effect relationships among historical emergency events and utilizing this knowledge to predict future emergencies. This approach involves Emergency Semantic Representation , Emergency Causality Inference , and Future Emergency Inference .

 
 

#### 4.6.1. Emergency Semantic Representation

 
 The first step in applying causality-based prediction to public emergencies is to extract relevant emergency events from various sources, including news articles, social media, and government reports, using natural language processing techniques. This process involves several approaches to represent emergency events in a meaningful way. One approach is the Event Phrase-based method, where the emergency event is represented as a phrase extracted from the text ( Barros et al., 2012 ) . For instance, if the text mentions a "terrorist attack in downtown," the phrase "terrorist attack in downtown" represents the event. Another approach is the Event Keywords-based method, where keywords extracted from the text are used to represent the emergency event ( Granroth-Wilding and Clark, 2016 ; Matsubara et al., 2012a ) . A third approach is the Tuple-based method ( Wang et al., 2015 ) , where each emergency event is represented by a tuple consisting of objects, a relationship, and time. For instance, if the text mentions a "car crash involving two vehicles on the highway at 4 pm," the tuple representation could be (car crash, involving, two vehicles, highway, 4 pm). An RDF-based format is utilized in some cases ( Cloke and Pappenberger, 2009 ) .
Mathematically, a tuple-based representation of an emergency event is expressed as E = ( O , R , T ) E=(O,R,T) . where O O represents the objects involved in the event, R R represents the relationship between the objects, and T T represents the time of the event. In the case of the car crash example mentioned earlier, the tuple representation is expressed as E = ( car crash, two vehicles, highway , involving , 4 pm ) E=(\text{car crash, two vehicles, highway},\text{involving},\text{4 pm}) 

 
 
 

#### 4.6.2. Emergency Causality Inference

 
 In this stage, the focus is on inferring cause-effect relationships among historical emergencies. The first step is to cluster the emergency events into emergency chains, sequences of time-ordered events with the same topics, actors, and objects ( Matsubara et al., 2012a ) . Once these chains are established, various approaches are used to infer the causal relationships among the emergency pairs. One common approach is to use natural language processing techniques to identify causal mentions, such as causal connectives, prepositions, and verbs ( Matsubara et al., 2012a ) , and then extract the causal relationships from the text. Another approach is to formulate causal-effect relationship identification as a classification task, where the inputs are the candidate events, and contextual information is incorporated, including related background knowledge from web texts ( Hu et al., 2017 ) .

 
 
 
 ∙ \bullet 
 
 Future Emergency Inference After learning the cause-effect relationships among historical emergency events and representing them semantically, they use this knowledge to predict future emergencies. This enables us to anticipate potential emergency situations and take preventative measures to mitigate their impact.

 

 ∙ \bullet 
 
 Retrieve similar emergencies To predict future emergencies, first search for similar events in the historical emergency event pool. They use various similarity measures, such as the Euclidean distance or cosine similarity, to compare the query event to historical events. Contextual information, such as the emergency time, location, and other relevant environmental and descriptive information, is taken into account. Specifically, use the following formula to calculate the Euclidean distance:

 

 
 (15) | 
 | 
 d ⁡ ( E 1 , E 2 ) = ∑ i = 1 n ( x 1 , i − x 2 , i ) 2 d(E_{1},E_{2})=\sqrt{\sum_{i=1}^{n}(x_{1,i}-x_{2,i})^{2}} | 
 | 
 

 where E 1 E_{1} and E 2 E_{2} are two emergency events being compared, x 1 , i x_{1,i} and x 2 , i x_{2,i} are the values of feature i i for E 1 E_{1} and E 2 E_{2} , respectively, and n n is the total number of features being compared.

 
 
 Similarly, use the cosine similarity to calculate the similarity between two emergency events:

 

 
 (16) | 
 | 
 s ​ i ​ m ​ ( E 1 , E 2 ) = ∑ i = 1 n x 1 , i ​ x 2 , i ∑ i = 1 n x 1 , i 2 ​ ∑ i = 1 n x 2 , i 2 sim(E_{1},E_{2})=\frac{\sum_{i=1}^{n}x_{1,i}x_{2,i}}{\sqrt{\sum_{i=1}^{n}x_{1,i}^{2}}\sqrt{\sum_{i=1}^{n}x_{2,i}^{2}}} | 
 | 
 

 where E 1 E_{1} and E 2 E_{2} are two emergency events being compared, x 1 , i x_{1,i} and x 2 , i x_{2,i} are the values of feature i i for E 1 E_{1} and E 2 E_{2} , respectively, and n n is the total number of features being compared. Then rank the retrieved emergency events based on their similarity to the query event. This approach allows us to identify potential future emergencies and take proactive steps to prevent or mitigate their impact.

 

 ∙ \bullet 
 
 Infer the future emergencies The next step is to determine the potential consequences caused by the query emergency based on the causality of emergencies learned from historical data. They traverse the abstraction tree that represents the learned causal relationships, starting from the root that corresponds to the most general emergency rule. The search frontier then moves across the tree if the child node is more similar, culminating in the nodes that are the least general but still similar to the new event being retrieved. Specifically, use the following formula to determine the similarity between two nodes in the abstraction tree:

 

 
 (17) | 
 | 
 s ​ i ​ m ​ ( n 1 , n 2 ) = w 1 , 2 w 1 , 1 ​ w 2 , 2 sim(n_{1},n_{2})=\frac{w_{1,2}}{\sqrt{w_{1,1}}\sqrt{w_{2,2}}} | 
 | 
 

 where n 1 n_{1} and n 2 n_{2} are two nodes being compared, w 1 , 2 w_{1,2} is the weight of the edge connecting nodes n 1 n_{1} and n 2 n_{2} , w 1 , 1 w_{1,1} is the sum of the weights of edges connected to node n 1 n_{1} , and w 2 , 2 w_{2,2} is the sum of the weights of edges connected to node n 2 n_{2} ( Catling and Wolff, 2019 ) .

 

 
 Since each case, event lead to multiple emergency events, use various approaches to determine the final prediction. For example, calculate the support or conditional probability of the rules using the following formula:

 

 
 (18) | 
 | 
 P ⁡ ( E | C ) = P ⁡ ( C | E ) ​ P ​ ( E ) P ⁡ ( C ) P(E|C)=\frac{P(C|E)P(E)}{P(C)} | 
 | 
 

 where P ⁡ ( E | C ) P(E|C) is the probability of emergency event E E given case event C C , P ⁡ ( C | E ) P(C|E) is the probability of case event C C given emergency event E E , P ⁡ ( E ) P(E) is the prior probability of emergency event E E , and P ⁡ ( C ) P(C) is the prior probability of case event C C .

 
 
 A ranking approach based on the similarity between the new emergency event and historical emergency events is employed. The similarity is defined by the length of their minimal generalization path, which is calculated using the following formula:

 

 
 (19) | 
 | 
 d ⁡ ( E 1 , E 2 ) = ∑ i = 1 k 1 2 i d(E_{1},E_{2})=\sum_{i=1}^{k}\frac{1}{2^{i}} | 
 | 
 

 
 
 where E 1 E_{1} and E 2 E_{2} are two emergency events being compared, and k k is the length of their minimal generalization path. The formula gives greater weight to earlier levels in the path, as the denominator increases exponentially with each level. This approach allows us to rank emergency events based on their similarity to the new event, and use this ranking to make a prediction ( Matsubara et al., 2012a ; Wang et al., 2015 ) .

 
 
 
 

### 4.7. Large Language Model Under Public Emergencies

 
 Large Language Models (LLMs) have great potential in addressing public emergencies, such as natural disasters and pandemics. LLMs provide real-time reports, predict outcomes, and analyze social media data to identify areas that are most affected. They also identify and address misinformation and fake news by analyzing social media data and news articles. Large Language Models, including the autoregressive integrated moving average (ARIMA) model, have demonstrated significant potential in predicting natural disasters through data analysis.

 
 
 In public emergency prediction, it would be predicting the outbreak of a disease based on the language used in social media posts. A formula that could be used in this category is the autoregressive integrated moving average (ARIMA) model:

 

 
 (20) | 
 | 
 Y t = c + ∑ i = 1 p ϕ i ​ Y t − i + ∑ i = 1 q θ i ​ ε t − i + ε t Y_{t}=c+\sum_{i=1}^{p}\phi_{i}Y_{t-i}+\sum_{i=1}^{q}\theta_{i}\varepsilon_{t-i}+\varepsilon_{t} | 
 | 
 

 where Y t Y_{t} is the number of social media posts at time t t , c c is a constant, ϕ i \phi_{i} and θ i \theta_{i} are the autoregressive and moving average parameters, respectively, and ε t \varepsilon_{t} is the error term at time t t .

 
 
 Then, it would be predicting the location and time of a flood based on historical data on weather patterns and water levels. A formula that could be used in this category is the space-time autoregressive integrated moving average (STARIMA) model:

 
 
 

 
 (21) | 
 | 
 Y s , t = c + ∑ i = 1 p ∑ j = 1 k ∑ l = 1 m ϕ i , j , l ​ Y s − i , t − j , l + ∑ i = 1 q ∑ j = 1 k ∑ l = 1 m θ i , j , l ​ ε s − i , t − j , l + ε s , t Y_{s,t}=c+\sum_{i=1}^{p}\sum_{j=1}^{k}\sum_{l=1}^{m}\phi_{i,j,l}Y_{s-i,t-j,l}+\sum_{i=1}^{q}\sum_{j=1}^{k}\sum_{l=1}^{m}\theta_{i,j,l}\varepsilon_{s-i,t-j,l}+\varepsilon_{s,t} | 
 | 
 

 where Y s , t Y_{s,t} is the water level at spatial location s s and time t t , ϕ i , j , l \phi_{i,j,l} and θ i , j , l \theta_{i,j,l} are the autoregressive and moving average parameters, respectively, and ε s , t \varepsilon_{s,t} is the error term at spatial location s s and time t t .

 
 
 After that, it would be predicting a wildfire’s location, time, and nature based on satellite imagery and textual data on weather patterns. A formula that could be used in this category is the space-time-text autoregressive model (STTAR) model:

 
 
 

 
 (22) | 
 | 
 Y s , t = c + ∑ i = 1 p ∑ j = 1 k ∑ l = 1 m ϕ i , j , l ​ Y s − i , t − j , l + ∑ i = 1 q ∑ j = 1 k ∑ l = 1 m θ i , j , l ​ ε s − i , t − j , l + β ​ X s , t + ε s , t Y_{s,t}=c+\sum_{i=1}^{p}\sum_{j=1}^{k}\sum_{l=1}^{m}\phi_{i,j,l}Y_{s-i,t-j,l}+\sum_{i=1}^{q}\sum_{j=1}^{k}\sum_{l=1}^{m}\theta_{i,j,l}\varepsilon_{s-i,t-j,l}+\beta X_{s,t}+\varepsilon_{s,t} | 
 | 
 

 where Y s , t Y_{s,t} is the severity of the wildfire at spatial location s s and time t t , ϕ i , j , l \phi_{i,j,l} and θ i , j , l \theta_{i,j,l} are the autoregressive and moving average parameters, respectively, X s , t X_{s,t} is a vector of weather and environmental variables at spatial location s s and time t t , β \beta is a vector of regression coefficients, and ε s , t \varepsilon_{s,t} is the error term at spatial location s s and time t t .

 
 
 In Vulnerability Prediction, it would be predicting a community’s vulnerability to a hurricane based on demographic and socio-economic data. A formula that could be used in this category is the vulnerability index:

 
 
 

 
 (23) | 
 | 
 V ​ I = ∑ i = 1 n w i ⋅ ( 1 − z i ) VI=\sum_{i=1}^{n}w_{i}\cdot(1-z_{i}) | 
 | 
 

 where w i w_{i} is the weight assigned to vulnerability factor i i , z i z_{i} is the standardized value of vulnerability factor i i , and n n is the number of vulnerability factors.

 
 
 In Association-based Impact Prediction, it would be predicting the impact of a hurricane based on its association with other factors, such as wind speed and storm surge ( Yan et al., 2022 ) . A formula that could be used in this category is the association rule ( Liu et al., 2022 ) : hurricane → damage \text{hurricane}\rightarrow\text{damage} , where the hurricane is the antecedent and the damage is the consequent. In Causality-based Prediction, it would be predicting the likelihood of a wildfire based on its causal factors, such as drought and lightning strikes ( Ding et al., 2021 ) . A formula that could be used in this category is the Bayesian network:

 
 
 

 
 (24) | 
 | 
 P ⁡ ( W | D , L ) = P ⁡ ( W ) ​ P ​ ( D , L | W ) P ⁡ ( D , L ) P(W|D,L)=\frac{P(W)P(D,L|W)}{P(D,L)} | 
 | 
 

 where W W is the event of a wildfire occurring, D D is the event of a drought occurring, L L is the event of lightning strikes occurring and P ⁡ ( W | D , L ) P(W|D,L) is the probability of a wildfire occurring given the occurrence of a drought and lightning strikes. P ⁡ ( W ) P(W) is the prior probability of a wildfire occurring, P ⁡ ( D , L | W ) P(D,L|W) is the conditional probability of a drought and lightning strikes occurring given that a wildfire has happened, and P ⁡ ( D , L ) P(D,L) is the joint probability of a drought and lightning strikes occurring.

 
 
 
 

## 5. Application of Public Emergencies Management

 
 This section introduces the applications of public emergency management. By dividing the applications of public emergency management into different phases, this section is divided into (1) Early Warning, (2) Disaster Monitoring, (3) Damage Assessment, and (4) Disaster Response.

 
 
 Table 7. Early warning 
 
 
 
 
 Category 
 | 
 
 
 Direction 
 | 
 
 
 Method 
 | 
 
 
 References 
 | 

 
 
 
 Early Warning 
 | 
 
 
 Sensing devices process data 
 | 
 
 
 PCA, Logistic Regression, CNN, RNN 
 | 
 
 
 ( Chin et al., 2020 ; He et al., 2023 ; Moon et al., 2019 ; Perol et al., 2018 ) 
 | 

 
 | 
 
 
 Sense the location of emergencies 
 | 
 
 
 GAN, RF, Image recognition 
 | 
 
 
 ( Huang and yang Xiang, 2018 ; Li et al., 2018d ; Lohumi and Roy, 2018 ; Weber et al., 2020 ) 
 | 

 
 | 
 
 
 Enhance the accuracy and speed 
 | 
 
 
 TLS, ANN, Fuzzy Deep Neural Network 
 | 
 
 
 ( Chen et al., 2017 ; Jiang et al., 2022 ; Zheng et al., 2017 ) 
 | 

 
 | 
 
 
 Utilize communication channels 
 | 
 
 
 Mining information dissemination data 
 | 
 
 
 ( Chen et al., 2017 ; Weber et al., 2020 ) 
 | 

 
 
 
 Disaster Monitoring 
 | 
 
 
 Element Sensing 
 | 
 
 
 UAV-based sensing 
 | 
 
 
 ( Cheng et al., 2020 ; Gopnarayan and Deshpande, 2020 ; Oliveira et al., 2016 ) 
 | 

 
 | 
 | 
 
 
 Visual sensing 
 | 
 
 
 ( Bang et al., 2019 ; Kil et al., 2019 ; Lopez-Cuevas et al., 2018 ) 
 | 

 
 | 
 
 
 Situation Sensing 
 | 
 
 
 Crowdsourced sensing 
 | 
 
 
 ( Poblete et al., 2018 ; Hsu et al., 2013 ) 
 | 

 
 | 
 | 
 
 
 Sensor Network-based System 
 | 
 
 
 ( Oliveira et al., 2016 ; Alam et al., 2018 ; Kil et al., 2019 ) 
 | 

 
 

### 5.1. Early Warning

 
 Early warning is a critical phase in public emergency management, aimed at detecting and predicting potential emergencies and providing timely alerts to relevant authorities and the public. To achieve early warning, various methods have been developed, including data mining, machine learning, and predictive modeling. For example, in the case of a flood, data from sources such as rainfall sensors, river level gauges, and weather forecasts are analyzed to predict the flood’s location, intensity, and possible impacts ( Inceoglu et al., 2018 ) . Early identification and warning of public emergencies are performed through Sensor Networks , Remote Sensing , Social Media , and other communication channels.

 
 

#### 5.1.1. Time Sensing

 
 Effective early warning of public emergencies relies on processing data acquired by sensing devices promptly. Moon et al. ( Moon et al., 2019 ) proposed a machine learning method for effective early warning of short-term rainfall, while He et al. ( He et al., 2023 ) utilized machine learning techniques to extract data from tweets and perform fusion analysis for rainstorm disasters in real-time. Perol et al. ( Perol et al., 2018 ) proposed a Convolutional Neural Network-based method for early detection and warning of earthquake disasters, and Chin et al. ( Chin et al., 2020 ) adopted a Recurrent Neural Network model for earthquake early warning systems. Zheng et al. ( Zheng et al., 2017 ) proposed a fuzzy deep neural network for early warning of industrial accidents.

 
 
 

#### 5.1.2. Location Sensing

 
 Early warning of public emergencies is critical in minimizing their impact. Seismic wave analysis has been used to sense the location of earthquakes, where Li et al. ( Li et al., 2018d ) used a Generative Adversarial Network. Ethan et al. ( Weber et al., 2020 ) performed image recognition on natural disaster images from social media to provide early warning of public emergencies such as earthquakes, floods, and wildfires. Machine learning models have also been used to sense the severity of flood events in videos. Huang et al. ( Huang and yang Xiang, 2018 ) proposed a deep belief network method for meteorological early warning of precipitation-induced landslides, achieving precise sensing of emergencies.

 
 
 Early warnings provide valuable information for governments, communities, and individuals to take appropriate actions to mitigate the impact of public emergencies. It also enables real-time sensing of conditions on land and sea using advanced computer numerical models. Ultimately, continuous improvement of early warning systems for public emergencies such as weather, climate, traffic, and epidemics is essential for effective disaster response.

 
 
 
 

### 5.2. Disaster Monitoring

 
 Continuous monitoring and assessment of public emergencies are crucial for providing emergency responders with crucial information to make informed decisions on response strategies. Disaster monitoring tools and technologies, such as Remote Sensing , Geographic Information Systems (GIS) , and Unmanned Aerial Vehicles (UAVs) , have been used to track and assess natural and man-made disasters.

 
 

#### 5.2.1. Element Sensing

 
 UAV-based sensing is a promising solution for sensing in public emergencies, with research focusing on detecting and monitoring gas leaks, floods, and forest fires ( Oliveira et al., 2016 ) . Sensor networks are used to collect data on gas concentrations, water levels, and temperature and humidity levels, which are transmitted to a central server for analysis, providing real-time information to emergency responders ( Cheng et al., 2020 ; Gopnarayan and Deshpande, 2020 ) . Visual sensing using VR and MR technologies, as well as crowdsourced sensing through social media platforms, are also valuable for sensing in public emergencies ( Bang et al., 2019 ; Doswell et al., 1993 ; Kang et al., 2017 ) . Advances in mobile technologies and social media have made it easier to collect data from citizens in affected areas, providing valuable information to emergency responders and improving the response time and effectiveness of emergency services ( Poblete et al., 2018 ) .

 
 
 

#### 5.2.2. Situation Understanding

 
 Table 8. Situation sensing 
 
 
 
 
 Category 
 | 
 
 
 Directions 
 | 
 
 
 Methods 
 | 
 
 
 References 
 | 

 
 | 
 
 
 Obtaining event data 
 | 
 
 
 Weakly supervised method 
 | 
 
 
 ( Yao et al., 2020 ; Wang et al., 2017 ; Alam et al., 2018 ) 
 | 

 
 | 
 
 
 Vulnerabilities in sensing devices 
 | 
 
 
 UAV-based sensing 
 | 
 
 
 ( Cheng et al., 2020 ; Oliveira et al., 2016 ) 
 | 

 
 
 
 Element sensing 
 | 
 
 
 Energy issues in fixed sensors 
 | 
 
 
 Crowdsourced sensing 
 | 
 
 
 ( Gopnarayan and Deshpande, 2020 ; Poblete et al., 2018 ) 
 | 

 
 | 
 
 
 Privacy issues in social media sensing 
 | 
 
 
 UAV-based sensing 
 | 
 
 
 ( Kil et al., 2019 ; Lopez-Cuevas et al., 2018 ) 
 | 

 
 | 
 
 
 Regional restrictions for sensing 
 | 
 
 
 Visual sensing 
 | 
 
 
 ( Doswell et al., 1993 ; Kang et al., 2017 ) 
 | 

 
 | 
 
 
 Efficient search for accurate information 
 | 
 
 
 Non-negative matrix factorization 
 | 
 
 
 ( Cheng et al., 2020 ; Lyu et al., 2019 ; Saldana et al., 2015 ; Gopnarayan and Deshpande, 2020 ) 
 | 

 
 | 
 
 
 Identifying sub-events 
 | 
 
 
 Unsupervised learning framework 
 | 
 
 
 ( Arachie et al., 2020 ; Wang et al., 2017 ; Oliveira et al., 2016 ) 
 | 

 
 
 
 Situational 
 | 
 
 
 Sensing data during GPS failures 
 | 
 
 
 Social media sensor monitoring 
 | 
 
 
 ( Hernandez-Suarez et al., 2019 ; Rashid et al., 2020 ; Bang et al., 2019 ) 
 | 

 
 
 
 understanding 
 | 
 
 
 Real-time image processing 
 | 
 
 
 Image processing pipeline 
 | 
 
 
 ( Alam et al., 2018 ; Chowdhury et al., 2020 ; Yu et al., 2020 ) 
 | 

 
 | 
 
 
 Combining heterogeneous data sources 
 | 
 
 
 Context-aware fusion method 
 | 
 
 
 ( Dao et al., 2018 ; Tijtgat et al., 2017 ; Lopez-Cuevas et al., 2018 ) 
 | 

 
 | 
 
 
 Transfer learning for COVID-19 
 | 
 
 
 Convolutional neural networks 
 | 
 
 
 ( Garg et al., 2020 ; Kil et al., 2019 ; Chowdhury et al., 2020 ; Saldana et al., 2015 ) 
 | 

 
 
 Situational understanding is critical in emergency response, as it improves the level of sensing and the depth of understanding, ultimately helping in managing public emergencies. However, situational sensing does not exist in isolation and must be assisted by element sensing to obtain event data.

 
 
 To deal with noisy data during public emergencies, weakly supervised methods and contextual messages are used to enrich message representations and achieve situational understanding. Cheng et al. ( Cheng et al., 2020 ) studied a novel topic-tracking problem and enabled efficient search for accurate information through an online non-negative matrix factorization scheme. Situation-sensing algorithms have been designed to automatically identify important sub-events during public emergencies. Social media sensors are useful for monitoring natural disasters and real-time understanding of image content, but filtering out irrelevant parts of images and combining heterogeneous data sources is required ( Yao et al., 2020 ; Arachie et al., 2020 ; Hernandez-Suarez et al., 2019 ; Alam et al., 2018 ; Dao et al., 2018 ) .

 
 
 
 

### 5.3. Damage Assessment

 
 Table 9. Damage Assessment and Disaster Response 
 
 
 
 
 Category 
 | 
 
 
 Directions 
 | 
 
 
 Methods 
 | 
 
 
 References 
 | 

 
 | 
 
 
 Social Media Analysis 
 | 
 
 
 DL-based multimodal approach 
 | 
 
 
 ( Wang et al., 2020c ; Nguyen et al., 2017b ) 
 | 

 
 | 
 
 
 Temporal and Spatial Analysis 
 | 
 
 
 Latent Dirichlet Allocation (LDA) 
 | 
 
 
 ( Resch et al., 2017a ; Scawthorn et al., 2006a ) 
 | 

 
 | 
 
 
 Aerial Image Analysis 
 | 
 
 
 Pre-trained DL CNN and ML 
 | 
 
 
 ( Yang and Cervone, 2019a ; Rizk et al., 2019 ) 
 | 

 
 
 
 Damage 
 | 
 
 
 Flood Damage Classification 
 | 
 
 
 K-means clustering and SVM 
 | 
 
 
 ( Akshya and Priyadarsini, 2019a ; Presa-Reyes and Chen, 2020a ) 
 | 

 
 
 
 Assessment 
 | 
 
 
 Damage Classification 
 | 
 
 
 CNN architecture 
 | 
 
 
 ( Presa-Reyes and Chen, 2020b ; Paul et al., 2020 ) 
 | 

 
 | 
 
 
 Deep CNN-based 
 | 
 
 
 Deep CNN 
 | 
 
 
 ( Dotel et al., 2020a ; Nguyen et al., 2017b ) 
 | 

 
 | 
 
 
 Multi-modal Classification 
 | 
 
 
 Two-stage multi-modal 
 | 
 
 
 ( Rizk et al., 2019 ; Kundu et al., 2018 ) 
 | 

 
 | 
 
 
 Damage Identification 
 | 
 
 
 CNNs and class activation maps 
 | 
 
 
 ( Li et al., 2018a ; Yang and Cervone, 2019b ) 
 | 

 
 
 
 Disaster Response 
 | 
 
 
 Public emergencies response 
 | 
 
 
 Drones, telemedicine, mobile apps 
 | 
 
 
 ( Jiang, 2019 ; Jiang et al., 2019 ; Jurgens, 2021 ) 
 | 

 
 | 
 
 
 Relief aid supply 
 | 
 
 
 Dynamic calculation 
 | 
 
 
 ( Wang et al., 2020c ; Zhao et al., 2015 ; Mutlu et al., 2019 ) 
 | 

 
 | 
 
 
 Social media data analysis 
 | 
 
 
 ML-based supervised models 
 | 
 
 
 ( Akshya and Priyadarsini, 2019b ; Nguyen et al., 2017c ; Resch et al., 2017b ) 
 | 

 
 | 
 
 
 Medical rescue 
 | 
 
 
 Decision table, genetic algorithm 
 | 
 
 
 ( Presa-Reyes and Chen, 2020a ; Scawthorn et al., 2006b ; Li et al., 2018b ) 
 | 

 
 | 
 
 
 Image classification 
 | 
 
 
 DL methods, ML algorithms 
 | 
 
 
 ( Yang and Cervone, 2019b ; Dotel et al., 2020b ; Peng et al., 2019 ) 
 | 

 
 | 
 
 
 Social media data classification 
 | 
 
 
 Adaptation classifiers, LSTM, CNN 
 | 
 
 
 ( Kabir and Madria, 2019 ; Kundu et al., 2018 ; Madichetty and Sridevi, 2019 ; Nguyen et al., 2017c ) 
 | 

 
 
 Damage assessment is crucial for effective emergency management. Techniques such as social media analysis and image processing provide real-time information to evaluate the extent of damage and plan resource deployment to mitigate the impact of the emergency on the affected population, infrastructure, and environment.

 
 
 Researchers have proposed various ML and DL techniques to minimize the impact of natural disasters. These techniques include analyzing social media data, aerial images, and satellite data to evaluate disaster situations and assess damage caused by natural disasters. Some of the techniques proposed include DL-based multimodal approach ( Wang et al., 2020c ) , ML techniques combined with temporal and spatial analysis of social media posts ( Resch et al., 2017a ) , hybrid approaches that classify areas affected by floods ( Akshya and Priyadarsini, 2019a ) , CNN architectures for damage classification ( Presa-Reyes and Chen, 2020b ; Rizk et al., 2019 ; Li et al., 2018a ) , and DL-based approaches to assess the impact of water-related disasters using satellite image data ( Dotel et al., 2020a ) . These techniques have shown promising results in identifying and assessing damage in disaster-hit areas. Overall, these proposed approaches using ML and DL techniques demonstrate their potential to improve disaster management and minimize the impact of natural disasters on the environment, infrastructure, and human lives. Damage assessment techniques provide accurate and timely information that is critical for planning and deploying resources to mitigate the impact of public emergencies.

 
 
 

### 5.4. Disaster Response

 
 Public emergency response is the phase of emergency management that involves the deployment of resources to mitigate the impact of the emergency and assist affected populations. This phase involves various activities, including medical assistance, evacuation, and infrastructure management.

 
 
 ML and DL techniques have been used to improve public emergency response and disaster management. These techniques include the use of MLP NN to estimate relief supplies ( Jiang, 2019 ) , decision tables to manage medical rescue ( Scawthorn et al., 2006b ) , and unmanned aerial vehicles for recognizing the status of individuals in disaster-struck areas ( Presa-Reyes and Chen, 2020a ) . ML algorithms have also been used to analyze social media data related to public emergencies ( Akshya and Priyadarsini, 2019b ; Nguyen et al., 2017c ; Li et al., 2018b ; Kundu et al., 2018 ; Madichetty and Sridevi, 2019 ; Kabir and Madria, 2019 ) and classify images from earthquake-hit areas ( Yang and Cervone, 2019b ) . These technologies have been tested on various datasets and for different types of disasters, aiding decision-making and response efforts. Technologies such as unmanned aerial vehicles, telemedicine, and mobile applications play a vital role in improving the efficiency and effectiveness of public emergency response and disaster management.

 
 
 
 

## 6. Lessons Learned and Future Work

 
 This study provides a comprehensive review of research papers focused on the prediction of public emergencies. The reviewed articles covered a broad range of research areas related to the use of social media data, machine learning models, and natural language processing techniques for analyzing emergency events and developing predictive models for different types of disasters. Despite significant progress in predicting information cascades during public emergencies, there are still several open questions and directions for future research to explore:

 
 

### 6.1. Complexity and Dynamics

 
 Future work in emergency response includes developing predictive algorithms that consider ethical and social implications. This requires using fairness, transparency, and accountability metrics to evaluate the impact on different populations, and creating ethical frameworks to guide AI use. Predictive algorithms must also account for complex and dynamic public emergencies, using advanced modeling techniques to capture interactions between emergency responders, the public, and stakeholders.

 
 
 

### 6.2. Model Predictability and Interpretability

 
 Predicting the popularity and spread of emergency information cascades is a challenging task, and there are fundamental questions that have yet to be answered. Additionally, it is crucial to understand the mechanisms that govern the success of emergency information dissemination ( Wang et al., 2020a ) . Interpretability of the predictions is also essential for building trust and accountability. However, these models are complex and challenging to interpret, which leads to issues with trust and accountability. Future research should focus on developing more interpretable models that are easily understood by human operators and provide a clear explanation of the reasoning behind their predictions.

 
 
 

### 6.3. Model Robustness to Noise and Adversarial Attacks

 
 Information cascades during public emergencies are easily influenced or manipulated by malicious actors, leading to inaccurate predictions and potentially harmful outcomes. External stimuli such as breaking news and rumors significantly affect the spread of emergency information. Modeling these external stimuli is important for improving the robustness of prediction models. Cross-domain real-time transfer learning and retrieval of information from other platforms are used to model external stimuli. Additionally, analyzing the sources of external stimuli provide insights into the future evolution of emergency information dissemination. Future research should aim to develop more robust models that handle noisy and adversarial input data and provide accurate predictions even in the presence of interference.

 
 
 

### 6.4. Human-AI teaming for public emergencies

 
 The field of Human-AI teaming has enormous potential in addressing public emergencies, such as natural disasters, pandemics, and terrorist attacks. This requires using fairness, transparency, and accountability metrics to evaluate the impact on different populations, and creating ethical frameworks to guide AI use. Predictive algorithms must also account for complex and dynamic public emergencies, using advanced modeling techniques to capture interactions between emergency responders, the public, and stakeholders.

 
 
 

### 6.5. Multimodal Large Models in Public Emergencies

 
 Future research should focus on developing multimodal, large-scale models that can effectively integrate and process diverse sources of information to generate accurate and timely predictions during sudden public events. These models should leverage deep learning techniques, such as natural language processing, computer vision, and speech recognition, to analyze different data modalities and identify patterns and trends. Incorporating contextual information and domain knowledge could further enhance their performance, ultimately improving our ability to anticipate and respond to sudden public events.

 
 
 
 

## 7. Conclusion

 
 This paper offers a comprehensive and systematic overview of existing techniques and methods for predicting emergency information cascades. The presented taxonomy serves as a valuable resource for domain experts when selecting the appropriate technique for a specific problem set. In particular, this paper analyzes the characteristics and methods of predicting information cascades during public emergencies from three perspectives: information cascade modeling, prediction, and application. These methods encompass a range of features and models, including time, location, and semantics, and incorporate the latest advances in modeling and predicting information cascades to provide a comprehensive and up-to-date overview of the field. Finally, we also discuss current research fronts, identify bottlenecks, pitfalls, and unresolved issues, and outline potential future research directions.

 
 
 

## References

 
 
 emd (2023) 
 
2023.

 
 EM-DAT.

 
 [EB/OL].

 
 
 
 https://www.emdat.be/ .

 

 
 Abuella and Chowdhury (2019) 
 
Mohamed Abuella and Badrul Chowdhury. 2019.

 
 Forecasting of solar power ramp events: A post-processing approach.

 
 Renewable Energy 133 (apr 2019), 1380–1392.

 
 
 https://doi.org/10.1016/j.renene.2018.09.005 

 

 
 Acharya et al . (2017) 
 
Saurav Acharya, Byung Suk Lee, and Paul Hines. 2017.

 
 Causal Prediction of Top-k Event Types Over Real-Time Event Streams.

 
 Comput. J. 60, 11 (feb 2017), 1561–1581.

 
 
 https://doi.org/10.1093/comjnl/bxw098 

 

 
 Adhikari et al . (2019) 
 
Bijaya Adhikari, Xinfeng Xu, Naren Ramakrishnan, and B. Aditya Prakash. 2019.

 
 EpiDeep: Exploiting Embeddings for Epidemic Forecasting (KDD ’19) . Association for Computing Machinery, New York, NY, USA, 577–586.

 
 

 https://doi.org/10.1145/3292500.3330917 

 

 
 Akshya and Priyadarsini (2019a) 
 
J. Akshya and P.L.K. Priyadarsini. 2019a.

 
 A Hybrid Machine Learning Approach for Classifying Aerial Images of Flood-Hit Areas. In 2019 International Conference on Computational Intelligence in Data Science (ICCIDS) . 1–5.

 
 
 https://doi.org/10.1109/ICCIDS.2019.8862138 

 

 
 Akshya and Priyadarsini (2019b) 
 
J. Akshya and P.L.K. Priyadarsini. 2019b.

 
 A Hybrid Machine Learning Approach for Classifying Aerial Images of Flood-Hit Areas. In 2019 International Conference on Computational Intelligence in Data Science (ICCIDS) . 1–5.

 
 
 https://doi.org/10.1109/ICCIDS.2019.8862138 

 

 
 Alam et al . (2018) 
 
Firoj Alam, Ferda Ofli, and Muhammad Imran. 2018.

 
 Processing Social Media Images by Combining Human and Machine Computing during Crises.

 
 International Journal of Human–Computer Interaction 34, 4 (jan 2018), 311–327.

 
 
 https://doi.org/10.1080/10447318.2018.1427831 

 

 
 Alevizos et al . (2017) 
 
Elias Alevizos, Alexander Artikis, and George Paliouras. 2017.

 
 Event Forecasting with Pattern Markov Chains. In Proceedings of the 11th ACM International Conference on Distributed and Event-Based Systems (Barcelona, Spain) (DEBS ’17) . Association for Computing Machinery, New York, NY, USA, 146–157.

 
 

 https://doi.org/10.1145/3093742.3093920 

 

 
 Alevizos et al . (2018) 
 
Elias Alevizos, Alexander Artikis, and Georgios Paliouras. 2018.

 
 Wayeb: a Tool for Complex Event Forecasting.

 
 
 https://doi.org/10.29007/2s9t 

 

 
 Allison (2018) 
 
Paul D Allison. 2018.

 
 Event history and survival analysis.

 
 In The reviewer’s guide to quantitative methods in the social sciences . Routledge, 86–97.

 
 
 

 
 Antunes et al . (2003) 
 
M. Antunes, M. A. Amaral Turkman, and K. F. Turkman. 2003.

 
 A Bayesian Approach to Event Prediction.

 
 Journal of Time Series Analysis 24, 6 (nov 2003), 631–646.

 
 
 https://doi.org/10.1111/j.1467-9892.2003.00326.x 

 

 
 Arachie et al . (2020) 
 
Chidubem Arachie, Manas Gaur, Sam Anzaroot, William Groves, Ke Zhang, and Alejandro Jaimes. 2020.

 
 Unsupervised Detection of Sub-Events in Large Scale Disasters.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 34, 01 (apr 2020), 354–361.

 
 
 https://doi.org/10.1609/aaai.v34i01.5370 

 

 
 Asim et al . (2016) 
 
K. M. Asim, F. Martínez-Álvarez, A. Basit, and T. Iqbal. 2016.

 
 Earthquake magnitude prediction in Hindukush region using machine learning techniques.

 
 Natural Hazards 85, 1 (sep 2016), 471–486.

 
 
 https://doi.org/10.1007/s11069-016-2579-3 

 

 
 Bang et al . (2019) 
 
Junseong Bang, Youngho Lee, Yong-Tae Lee, and Wonjoo Park. 2019.

 
 AR/VR Based Smart Policing For Fast Response to Crimes in Safe City. In 2019 IEEE International Symposium on Mixed and Augmented Reality Adjunct (ISMAR-Adjunct) . 470–475.

 
 
 https://doi.org/10.1109/ISMAR-Adjunct.2019.00126 

 

 
 Bao et al . (2019) 
 
Jie Bao, Pan Liu, and Satish V. Ukkusuri. 2019.

 
 A spatiotemporal deep learning approach for citywide short-term crash risk prediction with multi-source data.

 
 Accident Analysis Prevention 122 (jan 2019), 239–254.

 
 
 https://doi.org/10.1016/j.aap.2018.10.015 

 

 
 Bao et al . (2013) 
 
Peng Bao, Hua-Wei Shen, Junming Huang, and Xue-Qi Cheng. 2013.

 
 Popularity prediction in microblogging network: a case study on sina weibo. In Proceedings of the 22nd international conference on world wide web . 177–178.

 
 
 

 
 Barabasi (2005) 
 
Albert-Laszlo Barabasi. 2005.

 
 The origin of bursts and heavy tails in human dynamics.

 
 Nature 435, 7039 (2005), 207–211.

 
 
 

 
 Barnes et al . (2016) 
 
G. Barnes, K. D. Leka, C. J. Schrijver, T. Colak, R. Qahwaji, O. W. Ashamari, Y. Yuan, J. Zhang, R. T. J. McAteer, D. S. Bloomfield, P. A. Higgins, P. T. Gallagher, D. A. Falconer, M. K. Georgoulis, M. S. Wheatland, C. Balch, T. Dunn, and E. L. Wagner. 2016.

 
 A COMPARISON OF FLARE FORECASTING METHODS. I. RESULTS FROM THE “ALL-CLEAR” WORKSHOP.

 
 The Astrophysical Journal 829, 2 (sep 2016), 89.

 
 
 https://doi.org/10.3847/0004-637x/829/2/89 

 

 
 Barros et al . (2012) 
 
Vicente Barros, Christopher B. Field, Qin Dahe, and Thomas F. Stocker. 2012.

 
 Preface.

 
 In Managing the Risks of Extreme Events and Disasters to Advance Climate Change Adaptation . Cambridge University Press, ix–x.

 
 
 https://doi.org/10.1017/cbo9781139177245.002 

 

 
 Becker et al . (2012) 
 
Hila Becker, Dan Iter, Mor Naaman, and Luis Gravano. 2012.

 
 Identifying content for planned events across social media sites. In Proceedings of the fifth ACM international conference on Web search and data mining . ACM.

 
 
 https://doi.org/10.1145/2124295.2124360 

 

 
 Bhattacharjya et al . (2020) 
 
Debarun Bhattacharjya, Tian Gao, Nicholas Mattei, and Dharmashankar Subramanian. 2020.

 
 Cause-Effect Association between Event Pairs in Event Datasets. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence . International Joint Conferences on Artificial Intelligence Organization.

 
 
 https://doi.org/10.24963/ijcai.2020/167 

 

 
 Bialonski et al . (2015) 
 
Stephan Bialonski, Gerrit Ansmann, and Holger Kantz. 2015.

 
 Data-driven prediction and prevention of extreme events in a spatially extended excitable system.

 
 Physical Review E 92, 4 (oct 2015).

 
 
 https://doi.org/10.1103/physreve.92.042910 

 

 
 Bishop and Nasrabadi (2006) 
 
Christopher M Bishop and Nasser M Nasrabadi. 2006.

 
 Pattern recognition and machine learning . Vol. 4.

 
 Springer.

 
 
 

 
 Blair and Sambanis (2020) 
 
Robert A. Blair and Nicholas Sambanis. 2020.

 
 Forecasting Civil Wars: Theory and Structure in an Age of “Big Data” and Machine Learning.

 
 Journal of Conflict Resolution 64, 10 (apr 2020), 1885–1915.

 
 
 https://doi.org/10.1177/0022002720918923 

 

 
 Boni and Gerber (2016) 
 
Mohammad Al Boni and Matthew S. Gerber. 2016.

 
 Area-Specific Crime Prediction Models. In 2016 15th IEEE International Conference on Machine Learning and Applications (ICMLA) . IEEE.

 
 
 https://doi.org/10.1109/icmla.2016.0118 

 

 
 Brandt et al . (2011) 
 
Patrick T. Brandt, John R. Freeman, and Philip A. Schrodt. 2011.

 
 Real Time, Time Series Forecasting of Inter- and Intra-State Political Conflict.

 
 Conflict Management and Peace Science 28, 1 (feb 2011), 41–64.

 
 
 https://doi.org/10.1177/0738894210388125 

 

 
 Caigny et al . (2018) 
 
Arno De Caigny, Kristof Coussement, and Koen W. De Bock. 2018.

 
 A new hybrid classification algorithm for customer churn prediction based on logistic regression and decision trees.

 
 European Journal of Operational Research 269, 2 (sep 2018), 760–772.

 
 
 https://doi.org/10.1016/j.ejor.2018.02.009 

 

 
 Caigny et al . (2020) 
 
Arno De Caigny, Kristof Coussement, and Koen W. De Bock. 2020.

 
 Leveraging fine-grained transaction data for customer life event predictions.

 
 Decision Support Systems 130 (mar 2020), 113232.

 
 
 https://doi.org/10.1016/j.dss.2019.113232 

 

 
 Calabrese and Elkink (2016) 
 
Raffaella Calabrese and Johan A. Elkink. 2016.

 
 Estimating Binary Spatial Autoregressive Models for Rare Events.

 
 In Spatial Econometrics: Qualitative and Limited Dependent Variables . Emerald Group Publishing Limited, 145–166.

 
 
 https://doi.org/10.1108/s0731-905320160000037012 

 

 
 Casagrande et al . (2018) 
 
Flavia Dias Casagrande, Jim Torresen, and Evi Zouganeli. 2018.

 
 Sensor Event Prediction using Recurrent Neural Network in Smart Homes for Older Adults. In 2018 International Conference on Intelligent Systems (IS) . IEEE.

 
 
 https://doi.org/10.1109/is.2018.8710467 

 

 
 Catling and Wolff (2019) 
 
Finneas J R Catling and Anthony H Wolff. 2019.

 
 Temporal convolutional networks allow early prediction of events in critical care.

 
 Journal of the American Medical Informatics Association 27, 3 (dec 2019), 355–365.

 
 
 https://doi.org/10.1093/jamia/ocz205 

 

 
 Chan and Lam (2005) 
 
Ki Chan and Wai Lam. 2005.

 
 Extracting causation knowledge from natural language texts.

 
 International Journal of Intelligent Systems 20, 3 (2005), 327–358.

 
 
 https://doi.org/10.1002/int.20069 

 

 
 Chan and Franklin (2011) 
 
Samuel W.K. Chan and James Franklin. 2011.

 
 A text-based decision support system for financial sequence prediction.

 
 Decision Support Systems 52, 1 (dec 2011), 189–198.

 
 
 https://doi.org/10.1016/j.dss.2011.07.003 

 

 
 Chen et al . (2017) 
 
Jiaoyan Chen, Huajun Chen, Zhaohui Wu, Daning Hu, and Jeff Z. Pan. 2017.

 
 Forecasting smog-related health hazard based on social media and physical sensor.

 
 Information Systems 64 (mar 2017), 281–291.

 
 
 https://doi.org/10.1016/j.is.2016.03.011 

 

 
 Cheng et al . (2014) 
 
Justin Cheng, Lada Adamic, P Alex Dow, Jon Michael Kleinberg, and Jure Leskovec. 2014.

 
 Can cascades be predicted?. In Proceedings of the 23rd international conference on World wide web . 925–936.

 
 
 

 
 Cheng et al . (2020) 
 
Lu Cheng, Jundong Li, K. Selcuk Candan, and Huan Liu. 2020.

 
 Tracking Disaster Footprints with Social Streaming Data.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 34, 01 (apr 2020), 370–377.

 
 
 https://doi.org/10.1609/aaai.v34i01.5372 

 

 
 Chin et al . (2020) 
 
Tai-Lin Chin, Kuan-Yu Chen, Da-Yi Chen, and De-En Lin. 2020.

 
 Intelligent Real-Time Earthquake Detection by Recurrent Neural Networks.

 
 IEEE Transactions on Geoscience and Remote Sensing 58, 8 (aug 2020), 5440–5449.

 
 
 https://doi.org/10.1109/tgrs.2020.2966012 

 

 
 Choi et al . (2018) 
 
Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, and Jaegul Choo. 2018.

 
 StarGAN: Unified Generative Adversarial Networks for Multi-domain Image-to-Image Translation. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition . IEEE.

 
 
 https://doi.org/10.1109/cvpr.2018.00916 

 

 
 Chowdhury et al . (2020) 
 
Jishnu Ray Chowdhury, Cornelia Caragea, and Doina Caragea. 2020.

 
 On Identifying Hashtags in Disaster Twitter Data.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 34, 01 (apr 2020), 498–506.

 
 
 https://doi.org/10.1609/aaai.v34i01.5387 

 

 
 Cloke and Pappenberger (2009) 
 
H.L. Cloke and F. Pappenberger. 2009.

 
 Ensemble flood forecasting: A review.

 
 Journal of Hydrology 375, 3-4 (sep 2009), 613–626.

 
 
 https://doi.org/10.1016/j.jhydrol.2009.06.005 

 

 
 Coglianese and Nash (2016) 
 
Cary Coglianese and Jennifer Nash. 2016.

 
 Motivating without mandates? The role of voluntary programs in environmental governance.

 
 In Decision Making in Environmental Law . Edward Elgar Publishing, 237–252.

 
 
 https://doi.org/10.4337/9781783478408.ii.18 

 

 
 Compton et al . (2014) 
 
Ryan Compton, Craig Lee, Jiejun Xu, Luis Artieda-Moncada, Tsai-Ching Lu, Lalindra De Silva, and Michael Macy. 2014.

 
 Using publicly visible social media to build detailed forecasts of civil unrest.

 
 Security Informatics 3, 1 (sep 2014).

 
 
 https://doi.org/10.1186/s13388-014-0004-6 

 

 
 Cui et al . (2013) 
 
Peng Cui, Shifei Jin, Linyun Yu, Fei Wang, Wenwu Zhu, and Shiqiang Yang. 2013.

 
 Cascading outbreak prediction in networks: a data-driven approach. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining . 901–909.

 
 
 

 
 Damaschke et al . (2017) 
 
Magret Damaschke, Shane J. Cronin, and Mark S. Bebbington. 2017.

 
 A volcanic event forecasting model for multiple tephra records, demonstrated on Mt. Taranaki, New Zealand.

 
 Bulletin of Volcanology 80, 1 (12 2017).

 
 
 https://doi.org/10.1007/s00445-017-1184-y 

 

 
 Dao et al . (2018) 
 
Minh-Son Dao, Pham Quang Nhat Minh, Asem Kasem, and Mohamed Saleem Haja Nazmudeen. 2018.

 
 A Context-Aware Late-Fusion Approach for Disaster Image Retrieval from Social Media. In Proceedings of the 2018 ACM on International Conference on Multimedia Retrieval . ACM.

 
 
 https://doi.org/10.1145/3206025.3206047 

 

 
 Decroos et al . (2017) 
 
Tom Decroos, Vladimir Dzyuba, Jan Van Haaren, and Jesse Davis. 2017.

 
 Predicting Soccer Highlights from Spatio-Temporal Match Event Streams.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 31, 1 (feb 2017).

 
 
 https://doi.org/10.1609/aaai.v31i1.10754 

 

 
 Deep et al . (2020) 
 
Akash Deep, Dharmaraj Veeramani, and Shiyu Zhou. 2020.

 
 Event Prediction for Individual Unit Based on Recurrent Event Data Collected in Teleservice Systems.

 
 IEEE Transactions on Reliability 69, 1 (mar 2020), 216–227.

 
 
 https://doi.org/10.1109/tr.2019.2909471 

 

 
 Di et al . (2019) 
 
Xiaolei Di, Yu Xiao, Chao Zhu, Yang Deng, Qinpei Zhao, and Weixiong Rao. 2019.

 
 Traffic Congestion Prediction by Spatiotemporal Propagation Patterns. In 2019 20th IEEE International Conference on Mobile Data Management (MDM) . IEEE.

 
 
 https://doi.org/10.1109/mdm.2019.00-45 

 

 
 Ding et al . (2021) 
 
Yi Ding, Baoshen Guo, Lin Zheng, Mingming Lu, Desheng Zhang, Shuai Wang, Sang Hyuk Son, and Tian He. 2021.

 
 A City-Wide Crowdsourcing Delivery System with Reinforcement Learning.

 
 Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 5, 3 (sep 2021), 1–22.

 
 
 https://doi.org/10.1145/3478117 

 

 
 Ding et al . (2018) 
 
Zuohua Ding, Yuan Zhou, Geguang Pu, and MengChu Zhou. 2018.

 
 Online Failure Prediction for Railway Transportation Systems Based on Fuzzy Rules and Data Analysis.

 
 IEEE Transactions on Reliability 67, 3 (sep 2018), 1143–1158.

 
 
 https://doi.org/10.1109/tr.2018.2828113 

 

 
 Do et al . (2015) 
 
Quynh Ngoc Thi Do, Steven Bethard, and Marie-Francine Moens. 2015.

 
 Adapting Coreference Resolution for Narrative Processing. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing . Association for Computational Linguistics.

 
 
 https://doi.org/10.18653/v1/d15-1271 

 

 
 Doswell et al . (1993) 
 
Charles A. Doswell, Steven J. Weiss, and Robert H. Johns. 1993.

 
 Tornado forecasting: A review.

 
 In Geophysical Monograph Series . American Geophysical Union, 557–571.

 
 
 https://doi.org/10.1029/gm079p0557 

 

 
 Dotel et al . (2020a) 
 
Saramsha Dotel, Avishekh Shrestha, Anish Bhusal, Ramesh Pathak, Aman Shakya, and Sanjeeb Prasad Panday. 2020a.

 
 Disaster Assessment from Satellite Imagery by Analysing Topographical Features Using Deep Learning. In Proceedings of the 2020 2nd International Conference on Image, Video and Signal Processing . ACM.

 
 
 https://doi.org/10.1145/3388818.3389160 

 

 
 Dotel et al . (2020b) 
 
Saramsha Dotel, Avishekh Shrestha, Anish Bhusal, Ramesh Pathak, Aman Shakya, and Sanjeeb Prasad Panday. 2020b.

 
 Disaster Assessment from Satellite Imagery by Analysing Topographical Features Using Deep Learning. In Proceedings of the 2020 2nd International Conference on Image, Video and Signal Processing . ACM.

 
 
 https://doi.org/10.1145/3388818.3389160 

 

 
 Duan et al . (2020) 
 
Huilong Duan, Zhoujian Sun, Wei Dong, Kunlun He, and Zhengxing Huang. 2020.

 
 On Clinical Event Prediction in Patient Treatment Trajectory Using Longitudinal Electronic Health Records.

 
 IEEE Journal of Biomedical and Health Informatics 24, 7 (jul 2020), 2053–2063.

 
 
 https://doi.org/10.1109/jbhi.2019.2962079 

 

 
 ElRefai et al . (2022) 
 
Mohamed ElRefai, Mohamed Abouelasaad, Benedict M. Wiles, Anthony J. Dunn, Stefano Coniglio, Alain B. Zemkoho, and Paul R. Roberts. 2022.

 
 Deep learning-based insights on T:R ratio behaviour during prolonged screening for S-ICD eligibility.

 
 Journal of Interventional Cardiac Electrophysiology (may 2022).

 
 
 https://doi.org/10.1007/s10840-022-01245-6 

 

 
 Fang et al . (2020) 
 
Zhihan Fang, Guang Wang, Shuai Wang, Chaoji Zuo, Fan Zhang, and Desheng Zhang. 2020.

 
 CellRep: Usage Representativeness Modeling and Correction Based on Multiple City-Scale Cellular Networks. In Proceedings of The Web Conference 2020 . ACM.

 
 
 https://doi.org/10.1145/3366423.3380141 

 

 
 Fronza et al . (2013) 
 
Ilenia Fronza, Alberto Sillitti, Giancarlo Succi, Mikko Terho, and Jelena Vlasenko. 2013.

 
 Failure prediction based on log files using Random Indexing and Support Vector Machines.

 
 Journal of Systems and Software 86, 1 (jan 2013), 2–11.

 
 
 https://doi.org/10.1016/j.jss.2012.06.025 

 

 
 Fülöp et al . (2012) 
 
Lajos Jenő Fülöp, Árpád Beszédes, Gabriella Tóth, Hunor Demeter, László Vidács, and Lóránt Farkas. 2012.

 
 Predictive complex event processing. In Proceedings of the Fifth Balkan Conference in Informatics . ACM.

 
 
 https://doi.org/10.1145/2371316.2371323 

 

 
 Gallego-Castillo et al . (2015) 
 
Cristobal Gallego-Castillo, Alvaro Cuerva-Tejero, and Oscar Lopez-Garcia. 2015.

 
 A review on the recent history of wind power ramp forecasting.

 
 Renewable and Sustainable Energy Reviews 52 (dec 2015), 1148–1157.

 
 
 https://doi.org/10.1016/j.rser.2015.07.154 

 

 
 Galuba et al . (2010) 
 
Wojciech Galuba, Karl Aberer, Dipanjan Chakraborty, Zoran Despotovic, and Wolfgang Kellerer. 2010.

 
 Outtweeting the twitterers-predicting information cascades in microblogs.

 
 WOSN 10 (2010), 3–11.

 
 
 

 
 Gao et al . (2014a) 
 
Shuai Gao, Jun Ma, and Zhumin Chen. 2014a.

 
 Effective and effortless features for popularity prediction in microblogging network. In Proceedings of the 23rd International Conference on World Wide Web . 269–270.

 
 
 

 
 Gao et al . (2014b) 
 
Shuai Gao, Jun Ma, and Zhumin Chen. 2014b.

 
 Popularity prediction in microblogging network. In Web Technologies and Applications: 16th Asia-Pacific Web Conference, APWeb 2014, Changsha, China, September 5-7, 2014. Proceedings 16 . Springer, 379–390.

 
 
 

 
 Gao et al . (2019a) 
 
Xiaofeng Gao, Zhenhao Cao, Sha Li, Bin Yao, Guihai Chen, and Shaojie Tang. 2019a.

 
 Taxonomy and evaluation for microblog popularity prediction.

 
 ACM Transactions on Knowledge Discovery from Data (TKDD) 13, 2 (2019), 1–40.

 
 
 

 
 Gao and Zhao (2018) 
 
Yuyang Gao and Liang Zhao. 2018.

 
 Incomplete Label Multi-Task Ordinal Regression for Spatial Event Scale Forecasting.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 32, 1 (apr 2018).

 
 
 https://doi.org/10.1609/aaai.v32i1.11748 

 

 
 Gao et al . (2019b) 
 
Yuyang Gao, Liang Zhao, Lingfei Wu, Yanfang Ye, Hui Xiong, and Chaowei Yang. 2019b.

 
 Incomplete Label Multi-Task Deep Learning for Spatio-Temporal Event Subtype Forecasting.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 33, 01 (jul 2019), 3638–3646.

 
 
 https://doi.org/10.1609/aaai.v33i01.33013638 

 

 
 Garg et al . (2020) 
 
Tanmay Garg, Mamta Garg, Om Prakash Mahela, and Akhil Ranjan Garg. 2020.

 
 Convolutional Neural Networks with Transfer Learning for Recognition of COVID-19: A Comparative Study of Different Approaches.

 
 AI 1, 4 (dec 2020), 586–606.

 
 
 https://doi.org/10.3390/ai1040034 

 

 
 Ghil et al . (2011) 
 
M. Ghil, P. Yiou, S. Hallegatte, B. D. Malamud, P. Naveau, A. Soloviev, P. Friederichs, V. Keilis-Borok, D. Kondrashov, V. Kossobokov, O. Mestre, C. Nicolis, H. W. Rust, P. Shebalin, M. Vrac, A. Witt, and I. Zaliapin. 2011.

 
 Extreme events: dynamics, statistics and prediction.

 
 Nonlinear Processes in Geophysics 18, 3 (may 2011), 295–350.

 
 
 https://doi.org/10.5194/npg-18-295-2011 

 

 
 Gopnarayan and Deshpande (2020) 
 
Archana Gopnarayan and Sachin Deshpande. 2020.

 
 Tweets Analysis for Disaster Management: Preparedness, Emergency Response, Impact, and Recovery.

 
 In Innovative Data Communication Technologies and Application . Springer International Publishing, 760–764.

 
 
 https://doi.org/10.1007/978-3-030-38040-3_87 

 

 
 Granroth-Wilding and Clark (2016) 
 
Mark Granroth-Wilding and Stephen Clark. 2016.

 
 What Happens Next? Event Prediction Using a Compositional Neural Network Model.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 30, 1 (mar 2016).

 
 
 https://doi.org/10.1609/aaai.v30i1.10344 

 

 
 Gulmezoglu et al . (2019) 
 
Berk Gulmezoglu, Andreas Zankl, M. Caner Tol, Saad Islam, Thomas Eisenbarth, and Berk Sunar. 2019.

 
 Undermining User Privacy on Mobile Devices Using AI. In Proceedings of the 2019 ACM Asia Conference on Computer and Communications Security . ACM.

 
 
 https://doi.org/10.1145/3321705.3329804 

 

 
 Gürsun et al . (2011) 
 
Gonca Gürsun, Mark Crovella, and Ibrahim Matta. 2011.

 
 Describing and forecasting video access patterns. In 2011 proceedings IEEE infocom . IEEE, 16–20.

 
 
 

 
 Hagenau et al . (2012) 
 
Michael Hagenau, Michael Liebmann, Markus Hedwig, and Dirk Neumann. 2012.

 
 Automated News Reading: Stock Price Prediction Based on Financial News Using Context-Specific Features. In 2012 45th Hawaii International Conference on System Sciences . IEEE.

 
 
 https://doi.org/10.1109/hicss.2012.129 

 

 
 Han et al . (2012) 
 
Jiawei Han, Micheline Kamber, and Jian Pei. 2012.

 
 Data Preprocessing.

 
 In Data Mining . Elsevier, 83–124.

 
 
 https://doi.org/10.1016/b978-0-12-381479-1.00003-4 

 

 
 Hao et al . (2019) 
 
Mengmeng Hao, Dong Jiang, Fangyu Ding, Jingying Fu, and Shuai Chen. 2019.

 
 Simulating Spatio-Temporal Patterns of Terrorism Incidents on the Indochina Peninsula with GIS and the Random Forest Method.

 
 ISPRS International Journal of Geo-Information 8, 3 (mar 2019), 133.

 
 
 https://doi.org/10.3390/ijgi8030133 

 

 
 He et al . (2023) 
 
Jia He, Miao Ma, Yuxuan Zhou, and Miaoke Wang. 2023.

 
 What We Have Learned about the Characteristics and Differences of Disaster Information Behavior in Social Media—A Case Study of the 7.20 Henan Heavy Rain Flood Disaster.

 
 Sustainability 15, 6 (mar 2023), 4726.

 
 
 https://doi.org/10.3390/su15064726 

 

 
 Heaton (2017) 
 
Jeff Heaton. 2017.

 
 Ian Goodfellow, Yoshua Bengio, and Aaron Courville: Deep learning.

 
 Genetic Programming and Evolvable Machines 19, 1-2 (oct 2017), 305–307.

 
 
 https://doi.org/10.1007/s10710-017-9314-z 

 

 
 Hernandez-Suarez et al . (2019) 
 
Aldo Hernandez-Suarez, Gabriel Sanchez-Perez, Karina Toscano-Medina, Hector Perez-Meana, Jose Portillo-Portillo, Victor Sanchez, and Luis García Villalba. 2019.

 
 Using Twitter Data to Monitor Natural Disaster Social Dynamics: A Recurrent Neural Network Approach with Word Embeddings and Kernel Density Estimation.

 
 Sensors 19, 7 (apr 2019), 1746.

 
 
 https://doi.org/10.3390/s19071746 

 

 
 Hoegh et al . (2015) 
 
Andrew Hoegh, Scotland Leman, Parang Saraf, and Naren Ramakrishnan. 2015.

 
 Bayesian Model Fusion for Forecasting Civil Unrest.

 
 Technometrics 57, 3 (feb 2015), 332–340.

 
 
 https://doi.org/10.1080/00401706.2014.1001522 

 

 
 Hong et al . (2011) 
 
Liangjie Hong, Ovidiu Dan, and Brian D Davison. 2011.

 
 Predicting popular messages in twitter. In Proceedings of the 20th international conference companion on World wide web . 57–58.

 
 
 

 
 Hsu et al . (2013) 
 
Edbert B. Hsu, Yang Li, Jamil D. Bayram, David Levinson, Samuel Yang, and Colleen Monahan. 2013.

 
 State of Virtual Reality Based Disaster Preparedness and Response Training.

 
 PLoS Currents (2013).

 
 
 https://doi.org/10.1371/currents.dis.1ea2b2e71237d5337fa53982a38b2aff 

 

 
 Hu (2020) 
 
Linmei Hu. 2020.

 
 Integrating Hierarchical Attentions for Future Subevent Prediction.

 
 IEEE Access 8 (2020), 3106–3114.

 
 
 https://doi.org/10.1109/access.2019.2961973 

 

 
 Hu et al . (2017) 
 
Linmei Hu, Juanzi Li, Liqiang Nie, Xiao-Li Li, and Chao Shao. 2017.

 
 What Happens Next? Future Subevent Prediction Using Contextual Hierarchical LSTM.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 31, 1 (feb 2017).

 
 
 https://doi.org/10.1609/aaai.v31i1.11001 

 

 
 Huang and yang Xiang (2018) 
 
Lu Huang and Lu yang Xiang. 2018.

 
 Method for Meteorological Early Warning of Precipitation-Induced Landslides Based on Deep Neural Network.

 
 Neural Processing Letters 48, 2 (jan 2018), 1243–1260.

 
 
 https://doi.org/10.1007/s11063-017-9778-0 

 

 
 Hürriyetoǧlu et al . (2017) 
 
Ali Hürriyetoǧlu, Nelleke Oostdijk, and Antal van den Bosch. 2017.

 
 Estimating Time to Event of Future Events Based on Linguistic Cues on Twitter.

 
 In Intelligent Natural Language Processing: Trends and Applications . Springer International Publishing, 67–97.

 
 
 https://doi.org/10.1007/978-3-319-67056-0_5 

 

 
 Inceoglu et al . (2018) 
 
Fadil Inceoglu, Jacob H. Jeppesen, Peter Kongstad, Néstor J. Hernández Marcano, Rune H. Jacobsen, and Christoffer Karoff. 2018.

 
 Using Machine Learning Methods to Forecast if Solar Flares Will Be Associated with CMEs and SEPs.

 
 The Astrophysical Journal 861, 2 (jul 2018), 128.

 
 
 https://doi.org/10.3847/1538-4357/aac81e 

 

 
 Jiang et al . (2022) 
 
Nan Jiang, Hai-Bo Li, Cong-Jiang Li, Huai-Xian Xiao, and Jia-Wen Zhou. 2022.

 
 A Fusion Method Using Terrestrial Laser Scanning and Unmanned Aerial Vehicle Photogrammetry for Landslide Deformation Monitoring Under Complex Terrain Conditions.

 
 IEEE Transactions on Geoscience and Remote Sensing 60 (2022), 1–14.

 
 
 https://doi.org/10.1109/tgrs.2022.3181258 

 

 
 Jiang et al . (2019) 
 
Renhe Jiang, Xuan Song, Dou Huang, Xiaoya Song, Tianqi Xia, Zekun Cai, Zhaonan Wang, Kyoung-Sook Kim, and Ryosuke Shibasaki. 2019.

 
 DeepUrbanEvent. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery Data Mining . ACM.

 
 
 https://doi.org/10.1145/3292500.3330654 

 

 
 Jiang (2019) 
 
Zhe Jiang. 2019.

 
 A Survey on Spatial Prediction Methods.

 
 IEEE Transactions on Knowledge and Data Engineering 31, 9 (sep 2019), 1645–1664.

 
 
 https://doi.org/10.1109/tkde.2018.2866809 

 

 
 Jin et al . (2013) 
 
Fang Jin, Edward Dougherty, Parang Saraf, Yang Cao, and Naren Ramakrishnan. 2013.

 
 Epidemiological modeling of news and rumors on Twitter. In Proceedings of the 7th Workshop on Social Network Mining and Analysis . ACM.

 
 
 https://doi.org/10.1145/2501025.2501027 

 

 
 Jurgens (2021) 
 
David Jurgens. 2021.

 
 That's What Friends Are For: Inferring Location in Online Social Media Platforms Based on Social Relationships.

 
 Proceedings of the International AAAI Conference on Web and Social Media 7, 1 (aug 2021), 273–282.

 
 
 https://doi.org/10.1609/icwsm.v7i1.14399 

 

 
 Kabir and Madria (2019) 
 
Md. Yasin Kabir and Sanjay Madria. 2019.

 
 A Deep Learning Approach for Tweet Classification and Rescue Scheduling for Effective Disaster Management. In Proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems . ACM.

 
 
 https://doi.org/10.1145/3347146.3359097 

 

 
 Kang et al . (2017) 
 
Wei Kang, Jie Chen, Jiuyong Li, Jixue Liu, Lin Liu, Grant Osborne, Nick Lothian, Brenton Cooper, Terry Moschou, and Grant Neale. 2017.

 
 Carbon: Forecasting Civil Unrest Events by Monitoring News and Social Media.

 
 In Advanced Data Mining and Applications . Springer International Publishing, 859–865.

 
 
 https://doi.org/10.1007/978-3-319-69179-4_62 

 

 
 Kattan et al . (2015) 
 
Ahmed Kattan, Shaheen Fatima, and Muhammad Arif. 2015.

 
 Time-series event-based prediction: An unsupervised learning framework based on genetic programming.

 
 Information Sciences 301 (apr 2015), 99–123.

 
 
 https://doi.org/10.1016/j.ins.2014.12.054 

 

 
 Khoo et al . (2000) 
 
Christopher S. G. Khoo, Syin Chan, and Yun Niu. 2000.

 
 Extracting causal knowledge from a medical database using graphical patterns. In Proceedings of the 38th Annual Meeting on Association for Computational Linguistics - ACL '00 . Association for Computational Linguistics.

 
 
 https://doi.org/10.3115/1075218.1075261 

 

 
 Kil et al . (2019) 
 
Woogeun Kil, Kwangpyo Ko, Seungwoon Lee, and Byeong hee Roh. 2019.

 
 MR and IoT Convergence Platform with AI Support for Disaster Recognition (poster). In Proceedings of the 17th Annual International Conference on Mobile Systems, Applications, and Services . ACM.

 
 
 https://doi.org/10.1145/3307334.3328652 

 

 
 Kim (1993) 
 
Jaegwon Kim. 1993.

 
 Supervenience and Mind .

 
 Cambridge University Press.

 
 
 https://doi.org/10.1017/cbo9780511625220 

 

 
 Kleijnen and van Beers (2020) 
 
Jack P. C. Kleijnen and Wim C. M. van Beers. 2020.

 
 Prediction for Big Data Through Kriging: Small Sequential and One-Shot Designs.

 
 American Journal of Mathematical and Management Sciences 39, 3 (jan 2020), 199–213.

 
 
 https://doi.org/10.1080/01966324.2020.1716281 

 

 
 Kruengkrai et al . (2017) 
 
Canasai Kruengkrai, Kentaro Torisawa, Chikara Hashimoto, Julien Kloetzer, Jong-Hoon Oh, and Masahiro Tanaka. 2017.

 
 Improving Event Causality Recognition with Multiple Background Knowledge Sources Using Multi-Column Convolutional Neural Networks.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 31, 1 (feb 2017).

 
 
 https://doi.org/10.1609/aaai.v31i1.11005 

 

 
 Kulldorff (1997) 
 
Martin Kulldorff. 1997.

 
 A spatial scan statistic.

 
 Communications in Statistics - Theory and Methods 26, 6 (jan 1997), 1481–1496.

 
 
 https://doi.org/10.1080/03610929708831995 

 

 
 Kundu et al . (2018) 
 
Shamik Kundu, P.K Srijith, and Maunendra Sankar Desarkar. 2018.

 
 Classification of Short-Texts Generated During Disasters: A Deep Neural Network Based Approach. In 2018 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM) . IEEE.

 
 
 https://doi.org/10.1109/asonam.2018.8508695 

 

 
 Kunneman et al . (2020) 
 
F. Kunneman, M. van Mulken, and A. van den Bosch. 2020.

 
 Anticipointment Detection in Event Tweets.

 
 International Journal on Artificial Intelligence Tools 29, 02 (mar 2020), 2040001.

 
 
 https://doi.org/10.1142/s0218213020400011 

 

 
 Kupilik and Witmer (2018) 
 
Matthew Kupilik and Frank Witmer. 2018.

 
 Spatio-temporal violent event prediction using Gaussian process regression.

 
 Journal of Computational Social Science 1, 2 (aug 2018), 437–451.

 
 
 https://doi.org/10.1007/s42001-018-0024-y 

 

 
 Lakkaraju et al . (2013) 
 
Himabindu Lakkaraju, Julian McAuley, and Jure Leskovec. 2013.

 
 What’s in a name? understanding the interplay between titles, content, and communities in social media. In Proceedings of the international AAAI conference on web and social media , Vol. 7. 311–320.

 
 
 

 
 Laxman et al . (2008) 
 
Srivatsan Laxman, Vikram Tankasali, and Ryen W. White. 2008.

 
 Stream prediction using a generative model based on frequent episodes in event sequences. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining . ACM.

 
 
 https://doi.org/10.1145/1401890.1401947 

 

 
 Lei et al . (2019) 
 
Lei Lei, Xuguang Ren, Nigel Franciscus, Junhu Wang, and Bela Stantic. 2019.

 
 Event Prediction Based on Causality Reasoning.

 
 In Intelligent Information and Database Systems . Springer International Publishing, 165–176.

 
 
 https://doi.org/10.1007/978-3-030-14799-0_14 

 

 
 Leskovec et al . (2009) 
 
Jure Leskovec, Lars Backstrom, and Jon Kleinberg. 2009.

 
 Meme-tracking and the dynamics of the news cycle. In Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining . 497–506.

 
 
 

 
 Letham et al . (2013) 
 
Benjamin Letham, Cynthia Rudin, and David Madigan. 2013.

 
 Sequential event prediction.

 
 Machine Learning 93, 2-3 (jun 2013), 357–380.

 
 
 https://doi.org/10.1007/s10994-013-5356-5 

 

 
 Letham et al . (2015) 
 
Benjamin Letham, Cynthia Rudin, Tyler H. McCormick, and David Madigan. 2015.

 
 Interpretable classifiers using rules and Bayesian analysis: Building a better stroke prediction model.

 
 The Annals of Applied Statistics 9, 3 (sep 2015).

 
 
 https://doi.org/10.1214/15-aoas848 

 

 
 Li et al . (2016) 
 
Eldon Y. Li, Chen-Yuan Tung, and Shu-Hsun Chang. 2016.

 
 The wisdom of crowds in action: Forecasting epidemic diseases with a web-based prediction market system.

 
 International Journal of Medical Informatics 92 (aug 2016), 35–43.

 
 
 https://doi.org/10.1016/j.ijmedinf.2016.04.014 

 

 
 Li et al . (2017) 
 
Shengzhi Li, Jianzhong Qiao, and Shukuan Lin. 2017.

 
 Multi-attribute Event Modeling and Prediction over Event Streams from Sensors. In 2017 IEEE 23rd International Conference on Parallel and Distributed Systems (ICPADS) . IEEE.

 
 
 https://doi.org/10.1109/icpads.2017.00110 

 

 
 Li et al . (2018a) 
 
Xukun Li, Doina Caragea, Huaiyu Zhang, and Muhammad Imran. 2018a.

 
 Localizing and Quantifying Damage in Social Media Images. In 2018 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM) . IEEE.

 
 
 https://doi.org/10.1109/asonam.2018.8508298 

 

 
 Li et al . (2018b) 
 
Xukun Li, Doina Caragea, Huaiyu Zhang, and Muhammad Imran. 2018b.

 
 Localizing and Quantifying Damage in Social Media Images. In 2018 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM) . IEEE.

 
 
 https://doi.org/10.1109/asonam.2018.8508298 

 

 
 Li et al . (2018c) 
 
Zhongyang Li, Xiao Ding, and Ting Liu. 2018c.

 
 Constructing Narrative Event Evolutionary Graph for Script Event Prediction. In Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence . International Joint Conferences on Artificial Intelligence Organization.

 
 
 https://doi.org/10.24963/ijcai.2018/584 

 

 
 Li et al . (2018d) 
 
Zefeng Li, Men-Andrin Meier, Egill Hauksson, Zhongwen Zhan, and Jennifer Andrews. 2018d.

 
 Machine Learning Seismic Wave Discrimination: Application to Earthquake Early Warning.

 
 Geophysical Research Letters 45, 10 (may 2018), 4773–4779.

 
 
 https://doi.org/10.1029/2018gl077870 

 

 
 Li et al . (2007) 
 
Zhiguo Li, Shiyu Zhou, Suresh Choubey, and Crispian Sievenpiper. 2007.

 
 Failure event prediction using the Cox proportional hazard model driven by frequent failure signatures.

 
 IIE Transactions 39, 3 (mar 2007), 303–315.

 
 
 https://doi.org/10.1080/07408170600847168 

 

 
 Lin et al . (2019) 
 
Li Lin, Lijie Wen, and Jianmin Wang. 2019.

 
 MM-Pred: A Deep Predictive Model for Multi-attribute Event Sequence.

 
 In Proceedings of the 2019 SIAM International Conference on Data Mining . Society for Industrial and Applied Mathematics, 118–126.

 
 
 https://doi.org/10.1137/1.9781611975673.14 

 

 
 Lin et al . (2018) 
 
Ying-Lung Lin, Meng-Feng Yen, and Liang-Chih Yu. 2018.

 
 Grid-Based Crime Prediction Using Geographical Features.

 
 ISPRS International Journal of Geo-Information 7, 8 (jul 2018), 298.

 
 
 https://doi.org/10.3390/ijgi7080298 

 

 
 Liu et al . (2018) 
 
Bing Liu, Tong Yu, Ian Lane, and Ole Mengshoel. 2018.

 
 Customized Nonlinear Bandits for Online Response Selection in Neural Conversation Models.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 32, 1 (apr 2018).

 
 
 https://doi.org/10.1609/aaai.v32i1.12028 

 

 
 Liu and Brown (2004) 
 
H. Liu and D.E. Brown. 2004.

 
 A New Point Process Transition Density Model for Space–Time Event Prediction.

 
 IEEE Transactions on Systems, Man and Cybernetics, Part C (Applications and Reviews) 34, 3 (aug 2004), 310–324.

 
 
 https://doi.org/10.1109/tsmcc.2004.829306 

 

 
 Liu et al . (2022) 
 
Wei Liu, Yi Ding, Shuai Wang, Yu Yang, and Desheng Zhang. 2022.

 
 Para-Pred: Addressing Heterogeneity for City-Wide Indoor Status Estimation in On-Demand Delivery. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . ACM.

 
 
 https://doi.org/10.1145/3534678.3539167 

 

 
 Lohumi and Roy (2018) 
 
Kanishk Lohumi and Sudip Roy. 2018.

 
 Automatic Detection of Flood Severity Level from Flood Videos using Deep Learning Models. In 2018 5th International Conference on Information and Communication Technologies for Disaster Management (ICT-DM) . IEEE.

 
 
 https://doi.org/10.1109/ict-dm.2018.8636373 

 

 
 Lopez-Cuevas et al . (2018) 
 
Armando Lopez-Cuevas, Miguel Angel Medina-Perez, Raul Monroy, Jose Emmanuel Ramirez-Marquez, and Luis A. Trejo. 2018.

 
 FiToViz: A Visualisation Approach for Real-Time Risk Situation Awareness.

 
 IEEE Transactions on Affective Computing 9, 3 (jul 2018), 372–382.

 
 
 https://doi.org/10.1109/taffc.2017.2741478 

 

 
 Lu et al . (2017) 
 
Jiasen Lu, Caiming Xiong, Devi Parikh, and Richard Socher. 2017.

 
 Knowing When to Look: Adaptive Attention via a Visual Sentinel for Image Captioning. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) . IEEE.

 
 
 https://doi.org/10.1109/cvpr.2017.345 

 

 
 Lv et al . (2019) 
 
Shangwen Lv, Wanhui Qian, Longtao Huang, Jizhong Han, and Songlin Hu. 2019.

 
 SAM-Net: Integrating Event-Level and Chain-Level Attentions to Predict What Happens Next.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 33, 01 (jul 2019), 6802–6809.

 
 
 https://doi.org/10.1609/aaai.v33i01.33016802 

 

 
 Lyu et al . (2019) 
 
Dian Lyu, Peng Cheng, Ruizhou Liu, and Liang Liu. 2019.

 
 Bise-ResNet: Combine Segmentation and Classification Networks for Road Following on Unmanned Aerial Vehicle. In 2019 IEEE International Conference on Multimedia Expo Workshops (ICMEW) . IEEE.

 
 
 https://doi.org/10.1109/icmew.2019.00042 

 

 
 Ma and Leung (2019) 
 
King Ma and Henry Leung. 2019.

 
 A Novel LSTM Approach for Asynchronous Multivariate Time Series Prediction. In 2019 International Joint Conference on Neural Networks (IJCNN) . IEEE.

 
 
 https://doi.org/10.1109/ijcnn.2019.8851792 

 

 
 Ma et al . (2012) 
 
Zongyang Ma, Aixin Sun, and Gao Cong. 2012.

 
 Will this# hashtag be popular tomorrow?. In Proceedings of the 35th international ACM SIGIR conference on Research and development in information retrieval . 1173–1174.

 
 
 

 
 Ma et al . (2013) 
 
Zongyang Ma, Aixin Sun, and Gao Cong. 2013.

 
 On predicting the popularity of newly emerging hashtags in t witter.

 
 Journal of the American Society for Information Science and Technology 64, 7 (2013), 1399–1410.

 
 
 

 
 Madichetty and Sridevi (2019) 
 
Sreenivasulu Madichetty and M Sridevi. 2019.

 
 Detecting Informative Tweets during Disaster using Deep Neural Networks. In 2019 11th International Conference on Communication Systems Networks (COMSNETS) . IEEE.

 
 
 https://doi.org/10.1109/comsnets.2019.8711095 

 

 
 Mallouhy et al . (2019) 
 
Roxane Mallouhy, Chady Abou Jaoude, Christophe Guyeux, and Abdallah Makhoul. 2019.

 
 Major earthquake event prediction using various machine learning algorithms. In 2019 International Conference on Information and Communication Technologies for Disaster Management (ICT-DM) . IEEE.

 
 
 https://doi.org/10.1109/ict-dm47966.2019.9032983 

 

 
 Matsubara et al . (2012a) 
 
Yasuko Matsubara, Yasushi Sakurai, Christos Faloutsos, Tomoharu Iwata, and Masatoshi Yoshikawa. 2012a.

 
 Fast mining and forecasting of complex time-stamped events. In Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining . ACM.

 
 
 https://doi.org/10.1145/2339530.2339577 

 

 
 Matsubara et al . (2012b) 
 
Yasuko Matsubara, Yasushi Sakurai, B Aditya Prakash, Lei Li, and Christos Faloutsos. 2012b.

 
 Rise and fall patterns of information diffusion: model and implications. In Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining . 6–14.

 
 
 

 
 Minor and Cook (2017) 
 
Bryan Minor and Diane J. Cook. 2017.

 
 Forecasting occurrences of activities.

 
 Pervasive and Mobile Computing 38 (jul 2017), 77–91.

 
 
 https://doi.org/10.1016/j.pmcj.2016.09.010 

 

 
 Mirtaheri et al . (2019) 
 
Mehrnoosh Mirtaheri, Sami Abu-El-Haija, Fred Morstatter, Greg Ver Steeg, and Aram Galstyan. 2019.

 
 Identifying and Analyzing Cryptocurrency Manipulations in Social Media.

 
 (feb 2019).

 
 
 https://doi.org/10.31219/osf.io/dqz89 

 

 
 Moniz and Torgo (2019) 
 
Nuno Moniz and Luís Torgo. 2019.

 
 A review on web content popularity prediction: Issues and open challenges.

 
 Online Social Networks and Media 12 (2019), 1–20.

 
 
 

 
 Moon et al . (2019) 
 
Seung-Hyun Moon, Yong-Hyuk Kim, Yong Hee Lee, and Byung-Ro Moon. 2019.

 
 Application of machine learning to an early warning system for very short-term heavy rainfall.

 
 Journal of Hydrology 568 (jan 2019), 1042–1054.

 
 
 https://doi.org/10.1016/j.jhydrol.2018.11.060 

 

 
 Mukhina et al . (2019) 
 
Ksenia D. Mukhina, Alexander A. Visheratin, and Denis Nasonov. 2019.

 
 Urban events prediction via convolutional neural networks and Instagram data.

 
 Procedia Computer Science 156 (2019), 176–184.

 
 
 https://doi.org/10.1016/j.procs.2019.08.193 

 

 
 Muthiah et al . (2016) 
 
Sathappan Muthiah, Bert Huang, Jaime Arredondo, David Mares, Lise Getoor, Graham Katz, and Naren Ramakrishnan. 2016.

 
 Capturing Planned Protests from Open Source Indicators.

 
 AI Magazine 37, 2 (jul 2016), 63–75.

 
 
 https://doi.org/10.1609/aimag.v37i2.2631 

 

 
 Mutlu et al . (2019) 
 
Begum Mutlu, Hakan A. Nefeslioglu, Ebru A. Sezer, M. Ali Akcayol, and Candan Gokceoglu. 2019.

 
 An Experimental Research on the Use of Recurrent Neural Networks in Landslide Susceptibility Mapping.

 
 ISPRS International Journal of Geo-Information 8, 12 (dec 2019), 578.

 
 
 https://doi.org/10.3390/ijgi8120578 

 

 
 Nakajima et al . (2019) 
 
Yoko Nakajima, Keiya Takagi, Michal Ptaszynski, Hirotoshi Honma, and Fumito Masui. 2019.

 
 A Proposal of Prediction Method Using Word Polarity Information for Future Event Prediction Support System. In 2019 International Conference of Advanced Informatics: Concepts, Theory and Applications (ICAICTA) . IEEE.

 
 
 https://doi.org/10.1109/icaicta.2019.8904426 

 

 
 Naveed et al . (2011) 
 
Nasir Naveed, Thomas Gottron, Jérôme Kunegis, and Arifah Che Alhadi. 2011.

 
 Bad news travel fast: A content-based analysis of interestingness on twitter. In Proceedings of the 3rd international web science conference . 1–7.

 
 
 

 
 Nguyen et al . (2017a) 
 
Dai Quoc Nguyen, Dat Quoc Nguyen, Ashutosh Modi, Stefan Thater, and Manfred Pinkal. 2017a.

 
 A Mixture Model for Learning Multi-Sense Word Embeddings. In Proceedings of the 6th Joint Conference on Lexical and Computational Semantics (SEM 2017) . Association for Computational Linguistics.

 
 
 https://doi.org/10.18653/v1/s17-1015 

 

 
 Nguyen et al . (2017b) 
 
Dat T. Nguyen, Ferda Ofli, Muhammad Imran, and Prasenjit Mitra. 2017b.

 
 Damage Assessment from Social Media Imagery Data During Disasters. In Proceedings of the 2017 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining 2017 . ACM.

 
 
 https://doi.org/10.1145/3110025.3110109 

 

 
 Nguyen et al . (2017c) 
 
Dat T. Nguyen, Ferda Ofli, Muhammad Imran, and Prasenjit Mitra. 2017c.

 
 Damage Assessment from Social Media Imagery Data During Disasters. In Proceedings of the 2017 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining 2017 . ACM.

 
 
 https://doi.org/10.1145/3110025.3110109 

 

 
 Nsengiyumva and Valentino (2020) 
 
Jean Baptiste Nsengiyumva and Roberto Valentino. 2020.

 
 Predicting landslide susceptibility and risks using GIS-based machine learning simulations, case of upper Nyabarongo catchment.

 
 Geomatics, Natural Hazards and Risk 11, 1 (jan 2020), 1250–1277.

 
 
 https://doi.org/10.1080/19475705.2020.1785555 

 

 
 Ogie et al . (2018) 
 
Robert Ighodaro Ogie, Juan Castilla Rho, and Rodney J Clarke. 2018.

 
 Artificial intelligence in disaster risk communication: A systematic literature review. In 2018 5th International Conference on Information and Communication Technologies for Disaster Management (ICT-DM) . IEEE, 1–8.

 
 
 

 
 Okawa et al . (2019) 
 
Maya Okawa, Tomoharu Iwata, Takeshi Kurashima, Yusuke Tanaka, Hiroyuki Toda, and Naonori Ueda. 2019.

 
 Deep Mixture Point Processes. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery Data Mining . ACM.

 
 
 https://doi.org/10.1145/3292500.3330937 

 

 
 Oki et al . (2018) 
 
Motoyuki Oki, Koh Takeuchi, and Yukio Uematsu. 2018.

 
 Mobile Network Failure Event Detection and Forecasting With Multiple User Activity Data Sets.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 32, 1 (apr 2018).

 
 
 https://doi.org/10.1609/aaai.v32i1.11422 

 

 
 Oliveira et al . (2016) 
 
Gabriel L. Oliveira, Abhinav Valada, Claas Bollen, Wolfram Burgard, and Thomas Brox. 2016.

 
 Deep learning for human part discovery in images. In 2016 IEEE International Conference on Robotics and Automation (ICRA) . IEEE.

 
 
 https://doi.org/10.1109/icra.2016.7487304 

 

 
 Paul et al . (2020) 
 
Udit Paul, Alexander Ermakov, Michael Nekrasov, Vivek Adarsh, and Elizabeth Belding. 2020.

 
 #Outage: Detecting Power and Communication Outages from Social Networks. In Proceedings of The Web Conference 2020 . ACM.

 
 
 https://doi.org/10.1145/3366423.3380251 

 

 
 Peng et al . (2019) 
 
Bo Peng, Xinyi Liu, Zonglin Meng, and Qunying Huang. 2019.

 
 Urban Flood Mapping with Residual Patch Similarity Learning. In Proceedings of the 3rd ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery . ACM.

 
 
 https://doi.org/10.1145/3356471.3365235 

 

 
 Perol et al . (2018) 
 
Thibaut Perol, Michaël Gharbi, and Marine Denolle. 2018.

 
 Convolutional neural network for earthquake detection and location.

 
 Science Advances 4, 2 (feb 2018).

 
 
 https://doi.org/10.1126/sciadv.1700578 

 

 
 Petropoulos and Makridakis (2020) 
 
Fotios Petropoulos and Spyros Makridakis. 2020.

 
 Forecasting the novel coronavirus COVID-19.

 
 PLOS ONE 15, 3 (mar 2020), e0231236.

 
 
 https://doi.org/10.1371/journal.pone.0231236 

 

 
 Petrovic et al . (2011) 
 
Sasa Petrovic, Miles Osborne, and Victor Lavrenko. 2011.

 
 Rt to win! predicting message propagation in twitter. In Proceedings of the international AAAI conference on web and social media , Vol. 5. 586–589.

 
 
 

 
 Pillai et al . (2016) 
 
Karthik Ganesan Pillai, Rafal A. Angryk, Juan M. Banda, Dustin Kempton, Berkay Aydin, and Petrus C. Martens. 2016.

 
 Mining At Most Top-K Spatiotemporal Co-Occurrence Patterns in Datasets with Extended Spatial Representations.

 
 2, 3, Article 10 (sep 2016), 27 pages.

 
 

 https://doi.org/10.1145/2936775 

 

 
 Piraján et al . (2019) 
 
Freddy Piraján, Andrey Fajardo, and Miguel Melgarejo. 2019.

 
 Towards a Deep Learning Approach for Urban Crime Forecasting.

 
 In Communications in Computer and Information Science . Springer International Publishing, 179–189.

 
 
 https://doi.org/10.1007/978-3-030-31019-6_16 

 

 
 Poblete et al . (2018) 
 
Barbara Poblete, Jheser Guzman, Jazmine Maldonado, and Felipe Tobar. 2018.

 
 Robust Detection of Extreme Events Using Twitter: Worldwide Earthquake Monitoring.

 
 IEEE Transactions on Multimedia 20, 10 (oct 2018), 2551–2561.

 
 
 https://doi.org/10.1109/tmm.2018.2855107 

 

 
 Povinelli and Feng (2003) 
 
R.J. Povinelli and Xin Feng. 2003.

 
 A new temporal pattern identification method for characterization and prediction of complex time series events.

 
 IEEE Transactions on Knowledge and Data Engineering 15, 2 (mar 2003), 339–352.

 
 
 https://doi.org/10.1109/tkde.2003.1185838 

 

 
 Prasad et al . (2021) 
 
Pankaj Prasad, Victor Joseph Loveson, Bappa Das, and Mahender Kotha. 2021.

 
 Novel ensemble machine learning models in flood susceptibility mapping.

 
 Geocarto International 37, 16 (mar 2021), 4571–4593.

 
 
 https://doi.org/10.1080/10106049.2021.1892209 

 

 
 Presa-Reyes and Chen (2020a) 
 
Maria Presa-Reyes and Shu-Ching Chen. 2020a.

 
 Assessing Building Damage by Learning the Deep Feature Correspondence of Before and After Aerial Images. In 2020 IEEE Conference on Multimedia Information Processing and Retrieval (MIPR) . IEEE.

 
 
 https://doi.org/10.1109/mipr49039.2020.00017 

 

 
 Presa-Reyes and Chen (2020b) 
 
Maria Presa-Reyes and Shu-Ching Chen. 2020b.

 
 Assessing Building Damage by Learning the Deep Feature Correspondence of Before and After Aerial Images. In 2020 IEEE Conference on Multimedia Information Processing and Retrieval (MIPR) . IEEE.

 
 
 https://doi.org/10.1109/mipr49039.2020.00017 

 

 
 Qi et al . (2018) 
 
Xinshe Qi, Guo Li, Xin Wang, Na Wang, and Cuicui Gao. 2018.

 
 Predicting Model about Next Crime of Serial Offender. In 2018 5th International Conference on Information Science and Control Engineering (ICISCE) . IEEE.

 
 
 https://doi.org/10.1109/icisce.2018.00087 

 

 
 Radinsky et al . (2012) 
 
Kira Radinsky, Sagie Davidovich, and Shaul Markovitch. 2012.

 
 Learning causality for news events prediction. In Proceedings of the 21st international conference on World Wide Web . ACM.

 
 
 https://doi.org/10.1145/2187836.2187958 

 

 
 Ramakrishnan et al . (2014) 
 
Naren Ramakrishnan, Patrick Butler, Sathappan Muthiah, Nathan Self, Rupinder Khandpur, Parang Saraf, Wei Wang, Jose Cadena, Anil Vullikanti, Gizem Korkmaz, Chris Kuhlman, Achla Marathe, Liang Zhao, Ting Hua, Feng Chen, Chang Tien Lu, Bert Huang, Aravind Srinivasan, Khoa Trinh, Lise Getoor, Graham Katz, Andy Doyle, Chris Ackermann, Ilya Zavorin, Jim Ford, Kristen Summers, Youssef Fayed, Jaime Arredondo, Dipak Gupta, and David Mares.
2014.

 
 'Beating the news' with EMBERS. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining . ACM.

 
 
 https://doi.org/10.1145/2623330.2623373 

 

 
 Rashid et al . (2020) 
 
Md Tahmid Rashid, Daniel Yue Zhang, and Dong Wang. 2020.

 
 SocialDrone: An Integrated Social Media and Drone Sensing System for Reliable Disaster Response. In IEEE INFOCOM 2020 - IEEE Conference on Computer Communications . IEEE.

 
 
 https://doi.org/10.1109/infocom41043.2020.9155522 

 

 
 Rebane et al . (2019) 
 
Jonathan Rebane, Isak Karlsson, and Panagiotis Papapetrou. 2019.

 
 An Investigation of Interpretable Deep Learning for Adverse Drug Event Prediction. In 2019 IEEE 32nd International Symposium on Computer-Based Medical Systems (CBMS) . IEEE.

 
 
 https://doi.org/10.1109/cbms.2019.00075 

 

 
 Reid et al . (2018) 
 
David Reid, Abir Jaafar Hussain, Hissam Tawfik, Rozaida Ghazali, and Dhiya Al-Jumeily. 2018.

 
 Forecasting Natural Events Using Axonal Delay. In 2018 IEEE Congress on Evolutionary Computation (CEC) . IEEE.

 
 
 https://doi.org/10.1109/cec.2018.8477831 

 

 
 Ren et al . (2018) 
 
Honglei Ren, You Song, Jingwen Wang, Yucheng Hu, and Jinzhi Lei. 2018.

 
 A Deep Learning Approach to the Citywide Traffic Accident Risk Prediction. In 2018 21st International Conference on Intelligent Transportation Systems (ITSC) . IEEE.

 
 
 https://doi.org/10.1109/itsc.2018.8569437 

 

 
 Resch et al . (2017a) 
 
Bernd Resch, Florian Usländer, and Clemens Havas. 2017a.

 
 Combining machine-learning topic models and spatiotemporal analysis of social media data for disaster footprint and damage assessment.

 
 Cartography and Geographic Information Science 45, 4 (aug 2017), 362–376.

 
 
 https://doi.org/10.1080/15230406.2017.1356242 

 

 
 Resch et al . (2017b) 
 
Bernd Resch, Florian Usländer, and Clemens Havas. 2017b.

 
 Combining machine-learning topic models and spatiotemporal analysis of social media data for disaster footprint and damage assessment.

 
 Cartography and Geographic Information Science 45, 4 (aug 2017), 362–376.

 
 
 https://doi.org/10.1080/15230406.2017.1356242 

 

 
 Reyes et al . (2013) 
 
J. Reyes, A. Morales-Esteban, and F. Martínez-Álvarez. 2013.

 
 Neural networks to predict earthquakes in Chile.

 
 Applied Soft Computing 13, 2 (feb 2013), 1314–1328.

 
 
 https://doi.org/10.1016/j.asoc.2012.10.014 

 

 
 Ristea et al . (2020) 
 
Alina Ristea, Mohammad Al Boni, Bernd Resch, Matthew S. Gerber, and Michael Leitner. 2020.

 
 Spatial crime distribution and prediction for sporting events using social media.

 
 International Journal of Geographical Information Science 34, 9 (feb 2020), 1708–1739.

 
 
 https://doi.org/10.1080/13658816.2020.1719495 

 

 
 Rizk et al . (2019) 
 
Yara Rizk, Hadi Samer Jomaa, Mariette Awad, and Carlos Castillo. 2019.

 
 A computationally efficient multi-modal classification approach of disaster-related Twitter images. In Proceedings of the 34th ACM/SIGAPP Symposium on Applied Computing . ACM.

 
 
 https://doi.org/10.1145/3297280.3297481 

 

 
 Rostami et al . (2018) 
 
Mohammad Rostami, David Huber, and Tsai-Ching Lu. 2018.

 
 A crowdsourcing triage algorithm for geopolitical event forecasting. In Proceedings of the 12th ACM Conference on Recommender Systems . ACM.

 
 
 https://doi.org/10.1145/3240323.3240385 

 

 
 Rouet-Leduc et al . (2017) 
 
Bertrand Rouet-Leduc, Claudia Hulbert, Nicholas Lubbers, Kipton Barros, Colin J. Humphreys, and Paul A. Johnson. 2017.

 
 Machine Learning Predicts Laboratory Earthquakes.

 
 Geophysical Research Letters 44, 18 (sep 2017), 9276–9282.

 
 
 https://doi.org/10.1002/2017gl074677 

 

 
 Sakaki et al . (2010) 
 
Takeshi Sakaki, Makoto Okazaki, and Yutaka Matsuo. 2010.

 
 Earthquake shakes Twitter users. In Proceedings of the 19th international conference on World wide web . ACM.

 
 
 https://doi.org/10.1145/1772690.1772777 

 

 
 Saldana et al . (2015) 
 
David Saldana, Renato Assuncao, and Mario F. M. Campos. 2015.

 
 A distributed multi-robot approach for the detection and tracking of multiple dynamic anomalies. In 2015 IEEE International Conference on Robotics and Automation (ICRA) . IEEE.

 
 
 https://doi.org/10.1109/icra.2015.7139353 

 

 
 Salfner and Malek (2007) 
 
Felix Salfner and Miroslaw Malek. 2007.

 
 Using Hidden Semi-Markov Models for Effective Online Failure Prediction. In 2007 26th IEEE International Symposium on Reliable Distributed Systems (SRDS 2007) . IEEE.

 
 
 https://doi.org/10.1109/srds.2007.35 

 

 
 Santos et al . (2014) 
 
Raimundo Dos Santos, Sumit Shah, Feng Chen, Arnold Boedihardjo, Chang-Tien Lu, and Naren Ramakrishnan. 2014.

 
 Forecasting location-based events with spatio-temporal storytelling. In Proceedings of the 7th ACM SIGSPATIAL International Workshop on Location-Based Social Networks . ACM.

 
 
 https://doi.org/10.1145/2755492.2755496 

 

 
 Scawthorn et al . (2006a) 
 
Charles Scawthorn, Paul Flores, Neil Blais, Hope Seligson, Eric Tate, Stephanie Chang, Edward Mifflin, Will Thomas, James Murphy, Christopher Jones, and Michael Lawrence. 2006a.

 
 HAZUS-MH Flood Loss Estimation Methodology. II. Damage and Loss Assessment.

 
 Natural Hazards Review 7, 2 (may 2006), 72–81.

 
 
 https://doi.org/10.1061/(asce)1527-6988(2006)7:2(72) 

 

 
 Scawthorn et al . (2006b) 
 
Charles Scawthorn, Paul Flores, Neil Blais, Hope Seligson, Eric Tate, Stephanie Chang, Edward Mifflin, Will Thomas, James Murphy, Christopher Jones, and Michael Lawrence. 2006b.

 
 HAZUS-MH Flood Loss Estimation Methodology. II. Damage and Loss Assessment.

 
 Natural Hazards Review 7, 2 (may 2006), 72–81.

 
 
 https://doi.org/10.1061/(asce)1527-6988(2006)7:2(72) 

 

 
 Shao et al . (2017) 
 
Minglai Shao, Jianxin Li, Feng Chen, Hongyi Huang, Shuai Zhang, and Xunxun Chen. 2017.

 
 An Efficient Approach to Event Detection and Forecasting in Dynamic Multivariate Social Media Networks. In Proceedings of the 26th International Conference on World Wide Web . International World Wide Web Conferences Steering Committee.

 
 
 https://doi.org/10.1145/3038912.3052588 

 

 
 Shen et al . ([n. d.]) 
 
Yi-Dong Shen, Zhong Zhang, and Qiang Yang. [n. d.].

 
 Objective-oriented utility-based association mining. In 2002 IEEE International Conference on Data Mining, 2002. Proceedings. IEEE Comput. Soc.

 
 
 https://doi.org/10.1109/icdm.2002.1183938 

 

 
 Srinivasa et al . (2008) 
 
Narayan Srinivasa, Qin Jiang, and Leandro G. Barajas. 2008.

 
 High-Impact Event Prediction by Temporal Data Mining through Genetic Algorithms. In 2008 Fourth International Conference on Natural Computation . IEEE.

 
 
 https://doi.org/10.1109/icnc.2008.761 

 

 
 Su and Jiang (2020) 
 
Zichun Su and Jialin Jiang. 2020.

 
 Hierarchical Gated Recurrent Unit with Semantic Attention for Event Prediction.

 
 Future Internet 12, 2 (feb 2020), 39.

 
 
 https://doi.org/10.3390/fi12020039 

 

 
 Sun et al . (2020) 
 
Wenjuan Sun, Paolo Bocchini, and Brian D Davison. 2020.

 
 Applications of artificial intelligence for disaster management.

 
 Natural Hazards 103, 3 (2020), 2631–2689.

 
 
 

 
 Szabo and Huberman (2010) 
 
Gabor Szabo and Bernardo A Huberman. 2010.

 
 Predicting the popularity of online content.

 
 Commun. ACM 53, 8 (2010), 80–88.

 
 
 

 
 Tama and Comuzzi (2019) 
 
Bayu Adhi Tama and Marco Comuzzi. 2019.

 
 An empirical comparison of classification techniques for next event prediction using business process event logs.

 
 Expert Systems with Applications 129 (sep 2019), 233–245.

 
 
 https://doi.org/10.1016/j.eswa.2019.04.016 

 

 
 Tatar et al . (2014) 
 
Alexandru Tatar, Marcelo Dias De Amorim, Serge Fdida, and Panayotis Antoniadis. 2014.

 
 A survey on predicting the popularity of web content.

 
 Journal of Internet Services and Applications 5, 1 (2014), 1–20.

 
 
 

 
 Tatar et al . (2011) 
 
Alexandru Tatar, Jérémie Leguay, Panayotis Antoniadis, Arnaud Limbourg, Marcelo Dias de Amorim, and Serge Fdida. 2011.

 
 Predicting the popularity of online articles based on user comments. In Proceedings of the International Conference on Web Intelligence, Mining and Semantics . 1–8.

 
 
 

 
 Taylor (2017) 
 
James W. Taylor. 2017.

 
 Probabilistic forecasting of wind power ramp events using autoregressive logit models.

 
 European Journal of Operational Research 259, 2 (jun 2017), 703–712.

 
 
 https://doi.org/10.1016/j.ejor.2016.10.041 

 

 
 Tijtgat et al . (2017) 
 
Nils Tijtgat, Wiebe Van Ranst, Bruno Volckaert, Toon Goedeme, and Filip De Turck. 2017.

 
 Embedded Real-Time Object Detection for a UAV Warning System. In 2017 IEEE International Conference on Computer Vision Workshops (ICCVW) . IEEE.

 
 
 https://doi.org/10.1109/iccvw.2017.247 

 

 
 Tsagkias et al . (2010) 
 
Manos Tsagkias, Wouter Weerkamp, and Maarten De Rijke. 2010.

 
 News comments: Exploring, modeling, and online prediction. In Advances in Information Retrieval: 32nd European Conference on IR Research, ECIR 2010, Milton Keynes, UK, March 28-31, 2010. Proceedings 32 . Springer, 191–203.

 
 
 

 
 Vahedian et al . (2017) 
 
Amin Vahedian, Xun Zhou, Ling Tong, Yanhua Li, and Jun Luo. 2017.

 
 Forecasting Gathering Events through Continuous Destination Prediction on Big Trajectory Data. In Proceedings of the 25th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems . ACM.

 
 
 https://doi.org/10.1145/3139958.3140008 

 

 
 van Noord et al . (2017) 
 
Rik van Noord, Florian A. Kunneman, and Antal van den Bosch. 2017.

 
 Predicting Civil Unrest by Categorizing Dutch Twitter Events.

 
 In Communications in Computer and Information Science . Springer International Publishing, 3–16.

 
 
 https://doi.org/10.1007/978-3-319-67468-1_1 

 

 
 Vilalta and Ma ([n. d.]) 
 
R. Vilalta and Sheng Ma. [n. d.].

 
 Predicting rare events in temporal domains. In 2002 IEEE International Conference on Data Mining, 2002. Proceedings. IEEE Comput. Soc.

 
 
 https://doi.org/10.1109/icdm.2002.1183991 

 

 
 Wang et al . (2019b) 
 
Bao Wang, Penghang Yin, Andrea Louise Bertozzi, P. Jeffrey Brantingham, Stanley Joel Osher, and Jack Xin. 2019b.

 
 Deep Learning for Real-Time Crime Forecasting and Its Ternarization.

 
 Chinese Annals of Mathematics, Series B 40, 6 (nov 2019), 949–966.

 
 
 https://doi.org/10.1007/s11401-019-0168-y 

 

 
 Wang et al . (2017) 
 
Chen Wang, Hongzhi Lin, Rui Zhang, and Hongbo Jiang. 2017.

 
 SEND: A Situation-Aware Emergency Navigation Algorithm with Sensor Networks.

 
 IEEE Transactions on Mobile Computing 16, 4 (apr 2017), 1149–1162.

 
 
 https://doi.org/10.1109/tmc.2016.2582172 

 

 
 Wang et al . (2015) 
 
Chen Wang, Hoang Tam Vo, and Peng Ni. 2015.

 
 An IoT Application for Fault Diagnosis and Prediction. In 2015 IEEE International Conference on Data Science and Data Intensive Systems . IEEE.

 
 
 https://doi.org/10.1109/dsdis.2015.97 

 

 
 Wang and Ding (2015) 
 
Dawei Wang and Wei Ding. 2015.

 
 A Hierarchical Pattern Learning Framework for Forecasting Extreme Weather Events. In 2015 IEEE International Conference on Data Mining . IEEE.

 
 
 https://doi.org/10.1109/icdm.2015.93 

 

 
 Wang et al . (2013) 
 
Dawei Wang, Wei Ding, Kui Yu, Xindong Wu, Ping Chen, David L. Small, and Shafiqul Islam. 2013.

 
 Towards long-lead forecasting of extreme flood events. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining . ACM.

 
 
 https://doi.org/10.1145/2487575.2488220 

 

 
 Wang et al . (2020a) 
 
Guang Wang, Zhihan Fang, Xiaoyang Xie, Shuai Wang, Huijun Sun, Fan Zhang, Yunhuai Liu, and Desheng Zhang. 2020a.

 
 Pricing-aware Real-time Charging Scheduling and Charging Station Expansion for Large-scale Electric Buses.

 
 ACM Transactions on Intelligent Systems and Technology 12, 1 (nov 2020), 1–26.

 
 
 https://doi.org/10.1145/3428080 

 

 
 Wang et al . (2021a) 
 
Guang Wang, Zhou Qin, Shuai Wang, Huijun Sun, Zheng Dong, and Desheng Zhang. 2021a.

 
 Record: Joint Real-Time Repositioning and Charging for Electric Carsharing with Dynamic Deadlines. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . ACM.

 
 
 https://doi.org/10.1145/3447548.3467112 

 

 
 Wang et al . (2020d) 
 
Guang Wang, Harsh Rajkumar Vaish, Huijun Sun, Jianjun Wu, Shuai Wang, and Desheng Zhang. 2020d.

 
 Understanding User Behavior in Car Sharing Services Through The Lens of Mobility.

 
 Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 4, 4 (dec 2020), 1–30.

 
 
 https://doi.org/10.1145/3432200 

 

 
 Wang et al . (2020e) 
 
Guang Wang, Yongfeng Zhang, Zhihan Fang, Shuai Wang, Fan Zhang, and Desheng Zhang. 2020e.

 
 FairCharge:A Data-Driven Fairness-Aware Charging Recommendation System for Large-Scale Electric Taxi Fleets.

 
 Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 4, 1 (mar 2020), 1–25.

 
 
 https://doi.org/10.1145/3381003 

 

 
 Wang et al . (2021b) 
 
Guang Wang, Shuxin Zhong, Shuai Wang, Fei Miao, Zheng Dong, and Desheng Zhang. 2021b.

 
 Data-Driven Fairness-Aware Vehicle Displacement for Large-Scale Electric Taxi Fleets. In 2021 IEEE 37th International Conference on Data Engineering (ICDE) . IEEE.

 
 
 https://doi.org/10.1109/icde51399.2021.00108 

 

 
 Wang and Gerber (2015) 
 
Mingjun Wang and Matthew S. Gerber. 2015.

 
 Using Twitter for Next-Place Prediction, with an Application to Crime Prediction. In 2015 IEEE Symposium Series on Computational Intelligence . IEEE.

 
 
 https://doi.org/10.1109/ssci.2015.138 

 

 
 Wang et al . (2020b) 
 
Qi Wang, Guangyin Jin, Xia Zhao, Yanghe Feng, and Jincai Huang. 2020b.

 
 CSAN: A neural network benchmark model for crime forecasting in spatio-temporal scale.

 
 Knowledge-Based Systems 189 (feb 2020), 105120.

 
 
 https://doi.org/10.1016/j.knosys.2019.105120 

 

 
 Wang et al . (2019a) 
 
Shuai Wang, Tian He, Desheng Zhang, Yunhuai Liu, and Sang H. Son. 2019a.

 
 Towards Efficient Sharing: A Usage Balancing Mechanism for Bike Sharing Systems. In The World Wide Web Conference . ACM.

 
 
 https://doi.org/10.1145/3308558.3313441 

 

 
 Wang et al . (2020c) 
 
Tianyi Wang, Yudong Tao, Shu-Ching Chen, and Mei-Ling Shyu. 2020c.

 
 Multi-task Multimodal Learning for Disaster Situation Assessment. In 2020 IEEE Conference on Multimedia Information Processing and Retrieval (MIPR) . IEEE.

 
 
 https://doi.org/10.1109/mipr49039.2020.00050 

 

 
 Wang and Zhang (2017) 
 
Zhongqing Wang and Yue Zhang. 2017.

 
 DDoS Event Forecasting using Twitter Data. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence . International Joint Conferences on Artificial Intelligence Organization.

 
 
 https://doi.org/10.24963/ijcai.2017/580 

 

 
 Weber et al . (2020) 
 
Ethan Weber, Nuria Marzo, Dim P. Papadopoulos, Aritro Biswas, Agata Lapedriza, Ferda Ofli, Muhammad Imran, and Antonio Torralba. 2020.

 
 Detecting Natural Disasters, Damage, and Incidents in the Wild.

 
 In Computer Vision – ECCV 2020 . Springer International Publishing, 331–350.

 
 
 https://doi.org/10.1007/978-3-030-58529-7_20 

 

 
 Weiss and Page (2013) 
 
Jeremy C. Weiss and David Page. 2013.

 
 Forest-Based Point Process for Event Prediction from Electronic Health Records.

 
 In Advanced Information Systems Engineering . Springer Berlin Heidelberg, 547–562.

 
 
 https://doi.org/10.1007/978-3-642-40994-3_35 

 

 
 Wu et al . (2016) 
 
Bo Wu, Tao Mei, Wen-Huang Cheng, and Yongdong Zhang. 2016.

 
 Unfolding temporal dynamics: Predicting social media popularity using multi-scale temporal decomposition. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 30.

 
 
 

 
 Wu et al . (2019) 
 
Qitian Wu, Yirui Gao, Xiaofeng Gao, Paul Weng, and Guihai Chen. 2019.

 
 Dual sequential prediction models linking sequential recommendation and information dissemination. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery data mining . 447–457.

 
 
 

 
 Xiao et al . (2016) 
 
Shuai Xiao, Junchi Yan, Changsheng Li, Bo Jin, Xiangfeng Wang, Xiaokang Yang, Stephen M Chu, and Hongyuan Zha. 2016.

 
 On Modeling and Predicting Individual Paper Citation Count over Time.. In Ijcai . 2676–2682.

 
 
 

 
 Xiong et al . (2019) 
 
Chuanxiu Xiong, Ajitesh Srivastava, Rajgopal Kannan, Omkar Damle, Viktor Prasanna, and Erroll Southers. 2019.

 
 On Predicting Crime with Heterogeneous Spatial Patterns. In Proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems . ACM.

 
 
 https://doi.org/10.1145/3347146.3359374 

 

 
 Xue et al . (2018) 
 
Cong Xue, Zehua Zeng, Yuanye He, Lei Wang, and Neng Gao. 2018.

 
 A MIML-LSTM neural network for integrated fine-grained event forecasting. In Proceedings of 2018 International Conference on Big Data Technologies - ICBDT '18 . ACM Press.

 
 
 https://doi.org/10.1145/3226116.3226127 

 

 
 Yan et al . (2018) 
 
Hao Yan, Kamran Paynabar, and Jianjun Shi. 2018.

 
 Real-Time Monitoring of High-Dimensional Functional Data Streams via Spatio-Temporal Smooth Sparse Decomposition.

 
 Technometrics 60, 2 (apr 2018), 181–197.

 
 
 https://doi.org/10.1080/00401706.2017.1346522 

 

 
 Yan et al . (2022) 
 
Hua Yan, Shuai Wang, Yu Yang, Baoshen Guo, Tian He, and Desheng Zhang. 2022.

 
 $Oˆ { \{ 2 } \} $-SiteRec: Store Site Recommendation under the O2O Model via Multi-graph Attention Networks. In 2022 IEEE 38th International Conference on Data Engineering (ICDE) . IEEE.

 
 
 https://doi.org/10.1109/icde53745.2022.00044 

 

 
 Yang and Counts (2010) 
 
Jiang Yang and Scott Counts. 2010.

 
 Predicting the speed, scale, and range of information diffusion in twitter. In Proceedings of the International AAAI Conference on Web and Social Media , Vol. 4. 355–358.

 
 
 

 
 Yang and Leskovec (2011) 
 
Jaewon Yang and Jure Leskovec. 2011.

 
 Patterns of temporal variation in online media. In Proceedings of the fourth ACM international conference on Web search and data mining . 177–186.

 
 
 

 
 Yang et al . (2014) 
 
Jaewon Yang, Julian McAuley, Jure Leskovec, Paea LePendu, and Nigam Shah. 2014.

 
 Finding progression stages in time-evolving event sequences. In Proceedings of the 23rd international conference on World wide web . ACM.

 
 
 https://doi.org/10.1145/2566486.2568044 

 

 
 Yang and Cervone (2019a) 
 
Liping Yang and Guido Cervone. 2019a.

 
 Analysis of remote sensing imagery for disaster assessment using deep learning: a case study of flooding event.

 
 Soft Computing 23, 24 (mar 2019), 13393–13408.

 
 
 https://doi.org/10.1007/s00500-019-03878-8 

 

 
 Yang and Cervone (2019b) 
 
Liping Yang and Guido Cervone. 2019b.

 
 Analysis of remote sensing imagery for disaster assessment using deep learning: a case study of flooding event.

 
 Soft Computing 23, 24 (mar 2019), 13393–13408.

 
 
 https://doi.org/10.1007/s00500-019-03878-8 

 

 
 Yao et al . (2020) 
 
Wenlin Yao, Cheng Zhang, Shiva Saravanan, Ruihong Huang, and Ali Mostafavi. 2020.

 
 Weakly-Supervised Fine-Grained Event Recognition on Social Media Texts for Disaster Management.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 34, 01 (apr 2020), 532–539.

 
 
 https://doi.org/10.1609/aaai.v34i01.5391 

 

 
 Yi et al . (2019) 
 
Fei Yi, Zhiwen Yu, Fuzhen Zhuang, and Bin Guo. 2019.

 
 Neural Network based Continuous Conditional Random Field for Fine-grained Crime Prediction. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence . International Joint Conferences on Artificial Intelligence Organization.

 
 
 https://doi.org/10.24963/ijcai.2019/577 

 

 
 Yu et al . (2020) 
 
Manzhu Yu, Qunying Huang, Han Qin, Chris Scheele, and Chaowei Yang. 2020.

 
 Deep learning for real-time social media text classification for situation awareness – using Hurricanes Sandy, Harvey, and Irma as case studies.

 
 In Social Sensing and Big Data Computing for Disaster Management . Routledge, 33–50.

 
 
 https://doi.org/10.4324/9781003106494-3 

 

 
 Yu et al . (2019) 
 
Shuqi Yu, Linmei Hu, and Bin Wu. 2019.

 
 DRAM: A Deep Reinforced Intra-attentive Model for Event Prediction.

 
 In Knowledge Science, Engineering and Management . Springer International Publishing, 701–713.

 
 
 https://doi.org/10.1007/978-3-030-29551-6_62 

 

 
 Zafarani et al . (2014) 
 
Reza Zafarani, Mohammad Ali Abbasi, and Huan Liu. 2014.

 
 Social Media Mining .

 
 Cambridge University Press.

 
 
 https://doi.org/10.1017/cbo9781139088510 

 

 
 Zaman et al . (2014) 
 
Tauhid Zaman, Emily B Fox, and Eric T Bradlow. 2014.

 
 A bayesian approach for predicting the popularity of tweets.

 
 (2014).

 
 
 

 
 Zhang et al . (2016) 
 
Bolei Zhang, Zhuzhong Qian, and Sanglu Lu. 2016.

 
 Structure pattern analysis and cascade prediction in social networks. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2016, Riva del Garda, Italy, September 19-23, 2016, Proceedings, Part I 16 . Springer, 524–539.

 
 
 

 
 Zhao et al . (2016) 
 
Liang Zhao, Feng Chen, Chang-Tien Lu, and Naren Ramakrishnan. 2016.

 
 Multi-resolution Spatial Event Forecasting in Social Media. In 2016 IEEE 16th International Conference on Data Mining (ICDM) . IEEE.

 
 
 https://doi.org/10.1109/icdm.2016.0080 

 

 
 Zhao et al . (2015) 
 
Liang Zhao, Qian Sun, Jieping Ye, Feng Chen, Chang-Tien Lu, and Naren Ramakrishnan. 2015.

 
 Multi-Task Learning for Spatio-Temporal Event Forecasting. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining . ACM.

 
 
 https://doi.org/10.1145/2783258.2783377 

 

 
 Zhao et al . (2017) 
 
Sendong Zhao, Quan Wang, Sean Massung, Bing Qin, Ting Liu, Bin Wang, and ChengXiang Zhai. 2017.

 
 Constructing and Embedding Abstract Event Causality Networks from Text Snippets. In Proceedings of the Tenth ACM International Conference on Web Search and Data Mining . ACM.

 
 
 https://doi.org/10.1145/3018661.3018707 

 

 
 Zhao and Tang (2017) 
 
Xiangyu Zhao and Jiliang Tang. 2017.

 
 Modeling Temporal-Spatial Correlations for Crime Prediction. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management . ACM.

 
 
 https://doi.org/10.1145/3132847.3133024 

 

 
 Zheng et al . (2017) 
 
Yu-Jun Zheng, Sheng-Yong Chen, Yu Xue, and Jin-Yun Xue. 2017.

 
 A Pythagorean-Type Fuzzy Deep Denoising Autoencoder for Industrial Accident Early Warning.

 
 IEEE Transactions on Fuzzy Systems 25, 6 (dec 2017), 1561–1575.

 
 
 https://doi.org/10.1109/tfuzz.2017.2738605 

 

 
 Zhou et al . (2015) 
 
Cheng Zhou, Boris Cule, and Bart Goethals. 2015.

 
 A pattern based predictor for event streams.

 
 Expert Systems with Applications 42, 23 (dec 2015), 9294–9306.

 
 
 https://doi.org/10.1016/j.eswa.2015.08.021 

 

 
 Zhou et al . (2019) 
 
Lihua Zhou, Guowang Du, Ruxin Wang, Dapeng Tao, Lizhen Wang, Jun Cheng, and Jing Wang. 2019.

 
 A tensor framework for geosensor data forecasting of significant societal events.

 
 Pattern Recognition 88 (apr 2019), 27–37.

 
 
 https://doi.org/10.1016/j.patcog.2018.10.021