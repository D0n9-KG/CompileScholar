Survey of Computerized Adaptive Testing: A Machine Learning Perspective 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2404.00712v4 [cs.LG] 15 Mar 2026 
 
 

# Survey of Computerized Adaptive Testing: 
 A Machine Learning Perspective Thanks:  
Yan Zhuang, Qi Liu, Haoyang Bi, Zhenya Huang, Weizhe Huang, Jiatong Li, Junhao Yu, Zirui Liu, Zirui Hu, Yuting Hong, Mengxiao Zhu, and Enhong Chen are with State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China, China. Yan Zhuang is also with Nanjing University of Aeronautics and Astronautics, China. Zachary A. Pardos is with University of California, Berkeley, USA. Haiping Ma is with Anhui University, China. Shijin Wang is with iFLYTEK Co., Ltd, China.

 Corresponding E-mail: qiliuql@ustc.edu.cn
 

 
 
 Yan Zhuang
 
    
 Qi Liu
 
    
 Haoyang Bi
 
    
 Zhenya Huang
 
    
 Weizhe Huang
 
 Affiliation:  Jiatong Li, Junhao Yu, Zirui Liu, Zirui Hu, Yuting Hong, Zachary A. Pardos, Haiping Ma,
 
 Affiliation:  Mengxiao Zhu,  Shijin Wang, Enhong Chen, 

 

 Abstract 
 
 Computerized Adaptive Testing (CAT) offers an efficient and personalized method for assessing examinee proficiency by dynamically adjusting test questions based on individual performance. Compared to traditional, non-personalized testing methods, CAT requires fewer questions and provides more accurate assessments. As a result, CAT has been widely adopted across various fields, including education, healthcare, sports, sociology, and the evaluation of AI models. While traditional methods rely on psychometrics and statistics, the increasing complexity of large-scale testing has spurred the integration of machine learning techniques. This paper aims to provide a machine learning-focused survey on CAT, presenting a fresh perspective on this adaptive testing paradigm. We delve into measurement models, question selection algorithm, bank construction, and test control within CAT, exploring how machine learning can optimize these components. Through an analysis of current methods, strengths, limitations, and challenges, we strive to develop robust, fair, and efficient CAT systems. By bridging psychometric-driven CAT research with machine learning, this survey advocates for a more inclusive and interdisciplinary approach to the future of adaptive testing.

 
 
 
 Index Terms:  Adaptive testing, machine learning, proficiency assessment, AI evaluation, deep learning.

 
 

## I Introduction 

 
 The assessment of intelligent agents, whether human or AI systems, is essential for ensuring that individuals are well-prepared to meet the demands of their respective roles [ 1 , 2 ] . For humans, assessment results can determine eligibility for opportunities such as admissions or employment. For AI models, these results can indicate whether a system is suitable for deployment and capable of making real-world decisions. Traditionally, assessments have often used a one-size-fits-all approach, where all examinees answer the same set of questions, and a final score is calculated. Examples include traditional paper-and-pencil tests for humans and various gold-standard benchmarks for AI models.

 
 
 However, as the testing scale increases and the complexity and diversity of agents grow, traditional assessment methods face challenges in efficiency and reliability. Computerized Adaptive Testing (CAT), originating from psychometrics, offers a personalized testing paradigm by identifying and presenting the most informative and valuable questions to each examinee [ 3 , 4 ] . This method has been widely adopted in high-stakes testing scenarios for humans, such as the SAT, GRE, and GMAT [ 5 , 6 ] . Recently, CAT has also been increasingly used to assess AI’s capabilities, such as textual entailment recognition, chatbots, machine translation, and general-purpose AI systems [ 7 , 8 , 9 , 10 ] . CAT approach has been proved to require fewer questions to achieve the same level of assessment accuracy for both humans and AI systems [ 11 , 12 ] . Essentially, CAT aims to address a critical question about accuracy and efficiency : How to accurately estimate an examinee’s true proficiency while minimizing the number of questions provided?

 
 
 It is a dynamic and interactive process between an examinee (human or AI model) and a testing system. The testing system includes four main components that take turns: At each test step, the Measurement Model , as the user model , first uses the examinee’s previous responses to estimate their current proficiency, based on cognitive science or psychometrics [ 13 ] . Then, the Selection Algorithm picks the next question from the Question Bank according to certain criteria [ 14 , 15 , 16 ] . Most traditional criteria are statistical informativeness metrics, e.g., selecting the question whose difficulty matches the examinee’s current proficiency estimate, meaning the examinee has roughly a 50% chance of getting it right. The above process repeats until a predefined stopping rule is met. Throughout the assessment, Test Control governs various factors such as exposure balance, fairness, and robustness of the testing . At the conclusion of CAT, the final proficiency estimate—or diagnostic report—serves as the outcome of the assessment.

 
 
 CAT represents a complex fusion of machine intelligence and assessment techniques. It needs to manage large question banks, adapt to varying examinee proficiencies, and real-time decision-making. Moreover, practical CAT also involves ensuring reliability, fairness, search efficiency, etc. These challenges make CAT a multifaceted decision-making problem. 
With the rise of large-scale and diverse online testing platforms, these challenges have become even more significant. Machine learning (ML), particularly deep learning, offers promising solutions to enhance both the efficiency and accuracy of testing. Previous CAT surveys [ 17 , 4 , 18 , 19 ] have primarily focused on statistical and psychometric perspectives, concentrating mainly on human assessments. Given CAT’s interdisciplinary nature, this paper seeks to explore and review methodologies from a machine-learning perspective. It is more accessible to a broader readership and provides insights into building strong testing systems for both humans and artificial intelligence.

 
 
 In the realm of ML, CAT can be conceptualized as a parameter estimation problem with a focus on data efficiency [ 20 , 21 ] : The objective is to determine the values of latent parameters within a model (i.e., the examinee’s true proficiency) using the minimum amount of observed data (i.e., the fewest possible questions answered by the examinee). In recent years, there has been a growing interest in applying ML techniques to investigate the four components in CAT. For example, deep learning techniques diagnose examinee’s proficiency [ 22 ] and automate question bank construction [ 23 ] ; data-driven approaches optimize selection algorithms by learning from large-scale response data [ 24 , 25 , 26 ] . Despite these efforts, a comprehensive survey that captures the breadth of CAT solutions from a machine-learning perspective is still lacking. Furthermore, the ongoing evolution of machine learning presents new aspects for testing. The contributions of this paper are as follows:

 
 • 
 
 To our knowledge, this represents the first attempt to comprehensively review CAT solutions through the lens of machine learning. By exploring the existing work in Measurement Model, Selection Algorithms, Question Bank Construction, and Test Control, the paper offers a unified framework and encompasses the entire life cycle of the CAT system.

 

 • 
 
 We summarize existing works and draw conclusions on the success and failure attempts of machine learning. Furthermore, we identify key factors that are essential for building reliable and effective CAT systems for both human and AI model evaluation, including exposure control, fairness, robustness, and search efficiency. It offers a more comprehensive perspective. 

 

 • 
 
 We have open-sourced extensible and unified implementations of existing CAT models and relevant resources at https://github.com/bigdata-ustc/EduCAT . This library aims to assist researchers in swiftly developing a CAT system, encouraging collaboration, and ultimately leading to more sophisticated and effective CAT systems.

 

 
 The paper is organized as follows. In Section II , we introduce the background of CAT. Then in Section III , we provide the formulation of CAT’s task. After that, Section IV , V , and VI respectively review the existing methods for the measurement model, selection algorithms, and question bank construction. Given the fact that the selection algorithm is the core component for achieving the adaptivity, this survey mainly focuses on its recent advancements in machine learning and deep learning. In Section VII , we summarize the key factors in the application of CAT. Section VIII discusses how to evaluate the CAT.

 
 
 

## II Evolution of CAT 

 
 The evolution of CAT is a fascinating journey through time, marked by significant milestones. Adaptive testing began with Alfred Binet’s intelligence test in 1905 [ 27 ] . The 1950s saw the advent of computers, transforming adaptive testing into CAT. Key advancements in the 1970s and 1980s, particularly the integration of the psychometric model, enhanced assessment accuracy [ 28 , 29 ] . The 1990s internet boom made CAT widely accessible, leading to its use in major tests like the GRE, GMAT, and SAT. These tests, though evolved, still rely on adaptive principles. Various statistical methods optimize the testing experience, and CAT became a major focus in human measurement, covering education [ 30 , 6 , 31 ] , healthcare [ 32 , 33 , 34 ] , sociology [ 35 ] , and sports [ 36 , 37 ] .

 
 
 Recently, researchers have increasingly explored applying CAT to AI model evaluation. Existing benchmarks often contain redundant, low-quality, contaminated, or even erroneous questions, affecting the efficiency and reliability of AI assessments [ 11 ] . By leveraging adaptive testing, researchers can analyze the characteristics of benchmark questions to customize assessments for each AI system and estimate the latent traits behind each model’s responses, rather than merely calculating accuracy. Guided by CAT and psychometrics, various efficient methods have emerged in various aspects of AI evaluation, including performance estimation [ 8 , 38 ] , question selection [ 39 , 40 , 11 ] , and understanding experimental results [ 41 , 42 ] . These methods aim to identify informative and valuable subsets from large-scale datasets to improve the reliability of AI system evaluations.

 
 
 Current CAT research spans a wide range of topics, including the development of question banks, question selection, proficiency estimation, and various issues related to test security and reliability. They are critical to ensuring that CAT remains a reliable, valid, and fair assessment. Machine Learning is revolutionizing CAT by enabling sophisticated analysis of large datasets, detailed behavior modeling, and flexible adaptation to diverse testing environments [ 24 ] . Despite the ML in CAT is still in its early stages, its potential is evident. Machine learning offers new solutions to improve how we define, analyze, and apply CAT [ 43 ] . This survey aims to provide an overview and understanding of traditional statistical-based and recent ML-based CAT.

 
 
 Fig. 1: The workflow of CAT: At step t t , the selection algorithm adaptively selects next question q t + 1 q_{t+1} based on examinee’s current proficiency θ t {\theta}^{t} estimated by measurement models. 
 
 
 

## III Overview 

 
 An important assumption [ 17 ] of CAT is that examinee’s true proficiency level θ 0 ∈ ℝ d \theta_{0}\in\mathbb{R}^{d} is constant throughout the test. Here, d d represents the proficiency’s dimension; for example, θ \theta may correspond to a unidimensional overall ability level ( d = 1 d=1 ) or a multidimensional vector representing mastery levels across d d distinct knowledge concepts. The primary goal of CAT is to accurately and efficiently estimate examinees’ true proficiency levels by having them answer questions. Thus, CAT systems are designed to achieve two key objectives: (1) to use the responses to estimate an examinee’s proficiency θ \theta such that it closely approximates the true proficiency θ 0 \theta_{0} by the end of the test, and (2) to select the most valuable and fitting questions for each examinee, thereby reducing test length.

 
 

### III-A Task Formalization 

 
 To achieve the aforementioned objectives, CAT operates as an iterative and interactive process: As illustrated in Fig. 1 , at test step t ∈ [ 1 , 2 , … , T ] t\in[1,2,...,T] in CAT, examinee’s current proficiency estimate θ ^ t \hat{\theta}^{t} is estimated using previous t t responses; then leverage θ ^ t \hat{\theta}^{t} to retrieve the next question q t + 1 q_{t+1} from question bank 𝒬 \mathcal{Q} to ask examinee, and receive the next response label y t + 1 y_{t+1} . These interactions form a response sequence { ( q 1 , y 1 ) , ( q 2 , y 2 ) , … , ( q T , y T ) } \{(q_{1},y_{1}),(q_{2},y_{2}),...,(q_{T},y_{T})\} , where y t = 1 y_{t}=1 if the response to q t q_{t} is correct and 0 otherwise. To achieve the goals of CAT, each test step involves two critical processes:

 
 
 (1) Proficiency Estimation. The Measurement Model, denoted by f ⁡ ( ⋅ ) f(\cdot) , acts as a user model, predicting the probability of a correct response by an examinee with proficiency θ \theta , which is denoted as f ⁡ ( q , θ ) = P ⁡ ( y = 1 | q , θ ) f(q,\theta)=P(y=1|q,\theta) . The implementation of measurement model often draws upon cognitive science [ 44 ] or psychometrics [ 13 ] . To accurately estimate examinee’s proficiency at each step, various estimation methods can be used, e.g., Maximum Likelihood Estimation (MLE) or Bayesian Estimation. In applications, the binary cross-entropy loss is frequently utilized: at step t t , given previous t t responses 𝒟 1 : t = { ( q 1 , y 1 ) , ( q 2 , y 2 ) , … , ( q t , y t ) } \mathcal{D}_{1:t}=\{(q_{1},y_{1}),(q_{2},y_{2}),...,(q_{t},y_{t})\} , the corresponding empirical loss is:

 

 
 | 
 L ⁡ ( θ ) \displaystyle{L}(\theta) | 
 = ∑ ( q , y ) ∈ 𝒟 1 : t ℓ ( y , f ( q , θ ) ) \displaystyle=\sum_{(q,y)\in\mathcal{D}_{1:t}}\ell(y,f(q,\theta)) | 
 | 
 (1) | 
 
 
 | 
 | 
 = − ∑ ( q , y ) ∈ 𝒟 1 : t y log f ( q , θ ) + ( 1 − y ) log ⁡ ( 1 − f ⁡ ( q , θ ) ) , \displaystyle=-\sum_{(q,y)\in\mathcal{D}_{1:t}}{y\log f(q,\theta)+(1-y)\log(1-f(q,\theta))}, | 
 | 
 

 thus the current estimate of proficiency, θ ^ t \hat{\theta}^{t} , is obtained by minimizing the loss function L ⁡ ( θ ) L(\theta) : θ ^ t = arg ⁡ min θ ⁡ L ⁡ ( θ ) \hat{\theta}^{t}=\mathop{\arg\min}_{\theta}{{L}(\theta)} .

 
 
 (2) Question Selection. The heart of CAT is an algorithm that picks the next question q t + 1 q_{t+1} from the question bank 𝒬 \mathcal{Q} , using examinee’s current proficiency estimate θ ^ t \hat{\theta}^{t} as a guide:

 

 
 | 
 q t + 1 = arg ⁡ max q ∈ 𝒬 ⁡ 𝒱 q ​ ( θ ^ t ) , q_{t+1}=\mathop{\arg\max}_{q\in\mathcal{Q}}\mathcal{V}_{q}(\hat{\theta}^{t}), | 
 | 
 (2) | 
 

 where 𝒱 q ​ ( θ ^ t ) \mathcal{V}_{q}(\hat{\theta}^{t}) is the value of question q q . For instance, 𝒱 \mathcal{V} might be a measure of how much information the question will provide about the examinee’s proficiency, or it could be the output of a policy π \pi specifically designed to determine question selection.

 
 
 After receiving new response label y t + 1 y_{t+1} , measurement model updates and estimates proficiency θ ^ t + 1 \hat{\theta}^{t+1} . The above process will be repeated for T T times, ensuring the final step estimate θ ^ T \hat{\theta}^{T} close to the true θ 0 \theta_{0} , i.e.,

 
 
 Definition 1 (Definition of CAT) 
 
 The goal of CAT is to find a question set S = { q 1 , q 2 , … , q T } S=\{q_{1},q_{2},...,q_{T}\} of size T T , such that the final step estimate θ ^ T \hat{\theta}^{T} , derived from S S and their corresponding response labels y y , closely approximates the examinee’s true proficiency θ 0 \theta_{0} : 

 

 
 | 
 min | S | = T ⁡ ‖ θ ^ T − θ 0 ‖ . \min_{|S|=T}\|{\hat{\theta}^{T}}-\theta_{0}\|. | 
 | 
 (3) | 
 

 
 
 
 However, solving this optimization problem directly is impractical, as the true proficiency θ 0 \theta_{0} is not observable, and even the examinees may not know their exact proficiency level. Consequently, existing methods are all approximations of this target. For example, traditional statistical selection methods [ 14 , 15 ] utilize the asymptotic statistical properties of the MLE to reduce estimation uncertainty, e.g., selecting questions whose difficulty closely match the examinee’s current estimated proficiency θ ^ t \hat{\theta}^{t} . More recent Subset Selection approaches [ 21 ] try to identify a theoretical approximation of θ 0 \theta_{0} to serve as a new objective for optimization. For further details, refer to Section V .

 
 
 Meanwhile, as a practical system, considerations extend beyond proficiency estimation objective (Definition 1 ). Factors such as question exposure control, robustness, fairness, and search efficiency must be addressed as well. An exhaustive discussion of these factors is presented in Section VII .

 
 
 Evaluation Methods: To validate the accuracy of the estimated proficiency, two primary approaches are employed: 1) Performance prediction: using the examinee’s estimated values within the measurement model to predict the correctness label y y of the responses on examinee’s reserved response data, often measured by cross-entropy; 2) Proficiency estimation: using simulation to generate true proficiency values θ 0 \theta_{0} , simulating the examinee’s responses to each question. Then the Mean Squared Error (MSE) between the estimates and the simulated true values can be calculated. The details can be found in Section VIII .

 
 
 {forest} 
 
 Fig. 2: Summary of representative Computerized Adaptive Testing methods in machine learning perspective. 
 
 
 

### III-B Categorization 

 
 As shown in Fig. 2 , we categorize existing CAT research into four major components involved in the testing process described above: 1) Measurement Model, 2) Selection Algorithm, 3) Question Bank Construction, and 4) Test Control. Each part is further divided based on the different techniques employed. The following four sections (Sections  IV – VII ) provide detailed introductions and literature reviews of these components, with a particular focus on their theoretical foundations and methodological developments from a machine learning perspective. 

 
 
 
 

## IV Measurement Model 

 
 The existing methods for Measurement Model can be categorized into three main types: Item Response Theory (IRT), Cognitive Diagnostic Model (CDM), and Deep Learning Model.
In the first 50 years of CAT development, IRT was the dominant modeling framework and widely adopted in operational systems. It was not until 2009, with the introduction of CD-CAT [ 45 ] , that CDM began to be used as the underlying measurement model in adaptive testing. More recently, with the increasing scale and complexity of adaptive assessments and the rise of deep learning, a variety of new measurement models have emerged that go beyond traditional IRT and CDM. 

 
 
 Such classification is grounded on the nature of proficiency representation ( θ \theta ) in CAT—ranging from an overall numerical ability value (i.e., IRT), to discrete cognitive states across different knowledge concepts (i.e., CDM), to a unified modeling approach via deep learning techniques (i.e., Deep Learning Model). The choice of model should depend on the specific goals of the assessment, the nature of the data, and the resources available. Regardless of the chosen Measurement Model for estimating proficiency, the objective remains consistent: to minimize the error between the estimate and the true value at each step, expressed as ‖ θ ^ t − θ 0 ‖ → 0 \|{\hat{\theta}^{t}}-\theta_{0}\|\to 0 .

 
 

### IV-A Item Response Theory 

 
 In IRT [ 46 ] , an examinee’s proficiency is typically represented as a continuous scalar variable, referred to as overall ability. As a foundational framework in measurement models, IRT represents the examinee’s general level of proficiency using a latent trait parameter θ \theta . One of the most widely used models in IRT is the Three-Parameter Logistic Model (3PL-IRT). It utilizes a logistic-like interaction function to model the probability of examinee’s correct response to question j j , i.e.,

 

 
 | 
 f ⁡ ( q j , θ ) = c j + 1 − c j 1 + e − α j ​ ( θ − β j ) . f(q_{j},\theta)=c_{j}+\frac{1-c_{j}}{1+e^{-\alpha_{j}(\theta-\beta_{j})}}. | 
 | 
 (4) | 
 

 The 3PL-IRT model introduces three parameters ( β j , α j , c j \beta_{j},\alpha_{j},c_{j} ) for each test question j j : The difficulty parameter β j \beta_{j} corresponds to the level of proficiency at which an examinee has a 50% chance of answering the question correctly; The discrimination parameter α j \alpha_{j} describes how well the question differentiates between examinees with different ability; The guessing parameter c j c_{j} represents the probability that an examinee with a very low proficiency will answer the question correctly. In CAT systems, these parameters are pre-calibrated and remain fixed during the testing process, with annotation and calibration methods detailed in Section  VI .
IRT takes into account the number of questions answered correctly and the difficulty of the question. Almost all major adaptive tests for humans, such as SAT and GRE, are developed by using IRT, because the methodology can significantly improve measurement reliability and interpretability [ 47 ] . Recently, for AI system evaluation, Polo et al. [ 38 ] successfully selected 100 informative curated questions from MMLU [ 48 ] , a popular multiple-choice QA benchmark consisting of 14K questions, and accurately estimated the performance of LLMs.

 
 
 Multidimensional IRT (MIRT) [ 49 ] , on the other hand, extends IRT to multiple dimensions, allowing for the modeling of multiple latent traits simultaneously. Despite the great interpretability of (M)IRT models, their performance is constrained by the simplicity of the interaction function, and they lack fine-grained modeling about examinee’s cognitive states on individual knowledge concepts.

 
 
 

### IV-B Cognitive Diagnostic Model 

 
 Cognitive Diagnostic Model (CDM) is another representative class of measurement models, focusing on discrete knowledge concepts. Specifically, in CDMs, examinee proficiency is knowledge concept-wise and usually dichotomous , which indicates whether an examinee has mastered a knowledge concept or not. For example, in a mathematics assessment, knowledge concepts may include addition, fractions, or solving linear equations. We continue to use θ \theta to denote examinee proficiency for consistency. For example, the DINA method [ 44 , 50 ] models examinee proficiency θ = { θ ( 1 ) , θ ( 2 ) , … , θ ( K ) } \theta=\{\theta_{(1)},\theta_{(2)},...,\theta_{(K)}\} as their dichotomous knowledge mastery levels on all K K concepts. Given the Q-matrix Q ∈ ℝ | 𝒬 | × K Q\in\mathbb{R}^{|\mathcal{Q}|\times K} which is a binary matrix that indicates which knowledge concepts are associated with a question in bank 𝒬 \mathcal{Q} . DINA method focuses only on the knowledge concepts related to the target question j j , where Q j ​ k = 1 Q_{jk}=1 . Thus, the examinee’s binary response variable (with proficiency θ \theta ) to question j j is ∏ k , Q j ​ k = 1 θ ( k ) \prod_{k,Q_{jk}=1}\theta_{(k)} , and models questions as “slip” and “guess” parameters:

 

 
 | 
 f ⁡ ( q j , θ ) = ( 1 − s j ) ∏ k , Q j ​ k = 1 θ ( k ) ​ g j 1 − ∏ k , Q j ​ k = 1 θ ( k ) , f(q_{j},\theta)=(1-s_{j})^{\prod_{k,Q_{jk}=1}\theta_{(k)}}g_{j}^{1-\prod_{k,Q_{jk}=1}\theta_{(k)}}, | 
 | 
 (5) | 
 

 where s j s_{j} is the slip parameter, indicating the likelihood of an incorrect response despite mastery, and g j g_{j} is the guess parameter, reflecting the chance of a correct guess in the absence of mastery. Its extension G-DINA [ 51 ] provides a granular view of examinee proficiency, while FuzzyCDF [ 52 ] leverages fuzzy set theory for nuanced diagnostics from both objective and subjective data. Another approach, the Attribute Hierarchy Method [ 53 ] , applies rule space theory to structure knowledge dependencies and align examinee proficiencies with the nearest ideal cognitive patterns to obtain diagnostic results.

 
 
 Compared to IRT, CDM offers a more granular and comprehensive assessment of examinee proficiencies. They are particularly adept at providing detailed feedback on an individual’s strengths and weaknesses across multiple knowledge concepts, which is important for CAT and targeted further interventions. These models underscore a critical shift towards a more nuanced understanding of learning and proficiency, recognizing the multifaceted nature of knowledge acquisition and adaptive testing.

 
 
 

### IV-C Deep Learning Model 

 
 In recent years, the rapid growth of deep learning techniques stimulates the development of deep learning-driven Measurement Models. Compared to traditional models, deep learning methods are more suitable for measurements in large-scale data scenarios (e.g., online learning platforms) due to their efficiency and ability to learn the complex interaction pattern between examinees and questions.

 
 
 In these models, an examinee’s proficiency θ \theta is typically represented by a high-dimensional latent vector (embedding). Similarly, each question is encoded as an question embedding e j = Embed ​ ( q j ) e_{j}=\text{Embed}(q_{j}) . These embeddings are passed through a multi-layer neural network to predict the probability of a correct response: 

 

 
 | 
 f ( q j , θ ) = ϕ n ( ⋯ ϕ 1 ( W [ θ ; e j ] + b ) ⋯ ) , f(q_{j},\theta)=\phi_{n}\left(\cdots\phi_{1}\left(W[\theta;e_{j}]+b\right)\cdots\right), | 
 | 
 (6) | 
 

 where W W and b b are the weight matrix and bias vector, respectively, and ϕ k ​ ( ⋅ ) \phi_{k}(\cdot) denotes the activation function at the k k -th layer (e.g., ReLU, Tanh, or Sigmoid). 

 
 
 Based on this framework, several deep learning-based measurement models have demonstrated strong performance. For example, DIRT [ 54 ] uses a neural network to capture semantic information from question texts to empower accuracy. NeuralCD [ 22 ] utilizes a non-negative full connection neural network to capture the complex interaction, with the ability to generalize to other measurement models. Considering the complex heterogeneous relationships between examinees, questions, and knowledge concepts, massive efforts have also been made to leverage them to enhance measurements [ 55 , 56 , 57 ] .

 
 
 Discussion: In the CAT process, only a limited number of examinee responses can be obtained for proficiency estimation. To some extent, CAT can be viewed as a proficiency measurement under a cold start scenario . The performance of the measurement model is a critical factor in ensuring the accuracy of proficiency estimations within CAT. Meanwhile, it is important to note that the choice of the measurement model can significantly influence the selection of corresponding question selection algorithm.

 
 
 
 

## V Selection Algorithm 

 
 The selection algorithm is CAT’s core of implementing adaptivity and is the focal point of this survey. It utilizes the proficiency estimate obtained from the Measurement Model (introduced in the above section) to choose the next most suitable question, ensuring an accurate estimation of proficiency while using the fewest possible questions. Question selection algorithms can be categorized into traditional methods based on statistical information, as well as more recent machine learning methods, e.g., data-driven approaches (i.e., Reinforcement Learning and Meta Learning), and Subset Selection are becoming increasingly prevalent.

 
 

### V-A Statistical Algorithms 

 
 Generally, a practical approach to designing a selection algorithm involves developing quantitative methods to assign a numerical value to each question in the bank 𝒬 \mathcal{Q} . Classical statistical selection algorithms define the value of a question as the informativeness it provides about the examinee’s potential ability estimation. The next question index j t + 1 j_{t+1} can be selected from bank 𝒬 \mathcal{Q} based on current estimate θ ^ t \hat{\theta}^{t} :

 

 
 | 
 j t + 1 = arg ⁡ max q j ∈ 𝒬 ​ ℐ j ​ ( θ ^ t ) , j_{t+1}=\arg\max_{q_{j}\in\mathcal{Q}}\mathcal{I}_{j}(\hat{\theta}^{t}), | 
 | 
 (7) | 
 

 where ℐ j ​ ( ⋅ ) \mathcal{I}_{j}(\cdot) is the informativeness of question q j q_{j} (e.g., Fisher information). As illustrated in Definition 1 , CAT assumes that each examinee has a true proficiency value ( θ 0 \theta_{0} ) and it is considered as a parameter estimation process . The informativeness of an question can thus be interpreted as the expected contribution of the response on this question to the parameter estimation. This concept will be reflected in various selection algorithms discussed later.

 
 
 Fisher Information. In the parameter estimation problems, Fisher Information [ 58 ] is a concept from information theory and statistics that measures the amount of information that an observable random variable carries about the unknown parameter. In CAT, Fisher Information is often used to quantify the amount of information that a question provides about an examinee’s proficiency [ 18 ] . Specifically, we consider a random variable 𝒟 j = ( q j , y j ) \mathcal{D}_{j}=(q_{j},y_{j}) for which the pdf or pmf is f ⁡ ( q j , θ ) f(q_{j},\theta) , where θ \theta is the unknown parameter. The fisher info contained in the variable 𝒟 j \mathcal{D}_{j} is defined as: ℐ j ​ ( θ ) = 𝔼 y j ​ [ ( ∇ θ L ​ ( 𝒟 j | θ ) ) 2 ] = ( ∇ θ f ​ ( q j , θ ) ) 2 f ⁡ ( q j , θ ) ​ ( 1 − f ⁡ ( q j , θ ) ) \mathcal{I}_{j}(\theta)=\mathbb{E}_{y_{j}}[(\nabla_{\theta}L(\mathcal{D}_{j}|\theta))^{2}]=\frac{(\nabla_{\theta}f(q_{j},\theta))^{2}}{f(q_{j},\theta)(1-f(q_{j},\theta))} , where L ⁡ ( 𝒟 | θ ) = y ​ log ⁡ f ⁡ ( q , θ ) + ( 1 − y ) ​ log ⁡ ( 1 − f ⁡ ( q , θ ) ) L(\mathcal{D}|\theta)={y\log f(q,\theta)+(1-y)\log(1-f(q,\theta))} is the likelihood function of 𝒟 \mathcal{D} with respect to the parameter θ \theta .

 
 
 Thus, when using 3PL-IRT to model the f ⁡ ( q , θ ) f(q,\theta) and given current estimate θ ^ t \hat{\theta}^{t} , the Fisher Information of question j j can be calculated as:

 

 
 | 
 ℐ j ​ ( θ ^ t ) = ( 1 − c j ) ​ α j 2 ​ e − α j ​ ( θ ^ t − β j ) ( 1 + e − α j ​ ( θ ^ t − β j ) ) 2 ​ [ 1 − c j + c j ​ ( 1 + e − α j ​ ( θ ^ t − β j ) ) ] . \mathcal{I}_{j}(\hat{\theta}^{t})=\frac{(1-c_{j})\alpha_{j}^{2}e^{-\alpha_{j}(\hat{\theta}^{t}-\beta_{j})}}{(1+e^{-\alpha_{j}(\hat{\theta}^{t}-\beta_{j})})^{2}[1-c_{j}+c_{j}(1+e^{-\alpha_{j}(\hat{\theta}^{t}-\beta_{j})})]}. | 
 | 
 

 One crucial property of Fisher information is that its reciprocal (matrix inverse), is the variance (covariance matrix) of the asymptotic distribution of the proficiency estimate:

 
 
 Theorem 1 (The asymptotic distribution of MLE proficiency estimate [ 59 ] ) 
 
 At each step t t , based on the observation of examinee’s previous t t responses, the current proficiency estimate θ ^ t \hat{\theta}^{t} (estimated by MLE) satisfies the asymptotic normal distribution: θ ^ t ∼ 𝒩 ⁡ ( θ 0 , 1 t ​ ℐ ​ ( θ 0 ) ) . \hat{\theta}^{t}\sim\mathcal{N}\left(\theta_{0},\frac{1}{t\mathcal{I}(\theta_{0})}\right). 

 
 
 
 Obviously, as the number of questions t t or the Fisher Information ℐ ⁡ ( θ 0 ) \mathcal{I}(\theta_{0}) increases, the variance of the estimate decreases. Since θ ^ t \hat{\theta}^{t} is asymptotically unbiased (i.e., 𝔼 ⁡ [ θ ^ t ] = θ 0 \mathbb{E}[\hat{\theta}^{t}]=\theta_{0} ), a lower variance implies a more concentrated distribution around θ 0 \theta_{0} , thereby reducing estimation uncertainty and improving the estimation efficiency. 

 
 
 Fisher information has been popular in the development of personalized testing over the decades and extensively applied in various standardized human assessments. Similarly, for AI model evaluations, particularly for LLMs, the simple Fisher method allows for accurate performance estimation using only a small sample of test data. For example, Kipnis et al. [ 12 ] reduce six commonly used benchmarks to less than 3% of their original size while accurately estimating the performance of over 5,000 LLMs.

 
 
 When the measurement model is MIRT, Fisher information naturally extends from a scalar to a matrix [ 60 ] . Specifically, the information matrix provided by question j j at proficiency θ \theta (now a vector) is defined as: ℐ j ( θ ) = 𝔼 y j [ ∇ log L ( 𝒟 j | θ ) ∇ log L ( 𝒟 j | θ ) ⊤ ] \mathcal{I}_{j}(\theta)=\mathbb{E}_{y_{j}}\left[\nabla\log L(\mathcal{D}_{j}|\theta)\nabla\log L(\mathcal{D}_{j}|\theta)^{\top}\right] . This matrix’s inverse approximates the covariance of the MLE proficiency estimate. Based on this, various selection algorithms have been proposed, including D-Optimality (maximizing the determinant), A-Optimality (minimizing the trace of the inverse), and E-Optimality (maximizing the smallest eigenvalue) [ 49 ] .

 
 
 Kullback-Leibler Information. 
Fisher information is widely used in CAT for its theoretical foundation and mathematical simplicity. However, its effectiveness diminishes when the proficiency estimate deviates from the true value θ 0 \theta_{0} [ 17 ] . This issue becomes particularly evident in the early stages of a test when the estimate is still unstable due to limited responses. To address this issue, Chang et al. [ 15 ] proposed a global information measure based on Kullback–Leibler (KL) divergence. For the given question q j q_{j} (with response 𝒟 j \mathcal{D}_{j} ), the KL divergence between a candidate proficiency level θ \theta and the true proficiency θ 0 \theta_{0} is defined as:

 

 
 | 
 | 
 K L j ( θ ∥ θ 0 ) = 𝔼 y j log L ⁡ ( 𝒟 j | θ 0 ) L ⁡ ( 𝒟 j | θ ) \displaystyle KL_{j}(\theta\|{\theta}_{0})=\mathbb{E}_{y_{j}}\log\frac{L(\mathcal{D}_{j}|\theta_{0})}{L(\mathcal{D}_{j}|\theta)} | 
 | 
 
 
 | 
 = \displaystyle= | 
 f ⁡ ( q j , θ 0 ) ​ log ​ f ⁡ ( q j , θ 0 ) f ⁡ ( q j , θ ) + ( 1 − f ⁡ ( q j , θ 0 ) ) ​ log ​ 1 − f ⁡ ( q j , θ 0 ) 1 − f ⁡ ( q j , θ ) . \displaystyle f(q_{j},\theta_{0})\log\frac{f(q_{j},\theta_{0})}{f(q_{j},\theta)}+(1-f(q_{j},\theta_{0}))\log\frac{1-f(q_{j},\theta_{0})}{1-f(q_{j},\theta)}. | 
 | 
 

 
 
 The corresponding question selection algorithm integrates KL over a neighborhood of the current estimate θ ^ t \hat{\theta}^{t} :

 

 
 | 
 ℐ j ( θ ^ t ) = ∫ θ ^ t − δ θ ^ t + δ K L j ( θ | | θ ^ t ) d θ . \displaystyle\mathcal{I}_{j}(\hat{\theta}^{t})=\int_{\hat{\theta}^{t}-\delta}^{\hat{\theta}^{t}+\delta}KL_{j}(\theta||\hat{\theta}^{t})d\theta. | 
 | 
 (8) | 
 

 where δ = 3 / t \delta=3/\sqrt{t} . The integration range is wide at the beginning of the test and and gradually narrows as t t increases. In MIRT, this extends naturally to a multivariate integral. Essentially, the KL information identifies questions that will provide the greatest differentiation between the examinee’s possible proficiency levels . Unlike Fisher information, which depends on a single point estimate, KL information measures the discrepancy between two proficiency levels, θ \theta and θ 0 \theta_{0} , and remains effective even when they differ significantly. This is why KL information is global while Fisher information is local [ 17 ] . Specific examples comparing the two can be found in the appendix.

 
 
 Advanced Statistical Algorithms. Numerous works based on Fisher and KL information have been proposed. These methods try to introduce more information in selection to improve the efficiency of proficiency estimation. The Maximum Likelihood Weighted Information [ 61 ] weights the Fisher information by the likelihood function of the examinee’s current response results. Its rationale is similar to KL information and aims to improve the local limitations of Fisher information:
selecting the one that maximizes the integral of the likelihood function times the Fisher ℐ j ​ ( θ ) \mathcal{I}_{j}(\theta) over the proficiency level: ℐ j ( θ ^ t ) = ∫ θ ^ t − δ θ ^ t + δ L ( 𝒟 1 : t − 1 | θ ) ℐ j ( θ ) d θ \mathcal{I}_{j}(\hat{\theta}^{t})=\int_{\hat{\theta}^{t}-\delta}^{\hat{\theta}^{t}+\delta}L(\mathcal{D}_{1:t-1}|\theta)\mathcal{I}_{j}(\theta)d\theta , where L ( 𝒟 1 : t − 1 | θ ) = ∑ j = 1 t − 1 L ( 𝒟 j | θ ) L(\mathcal{D}_{1:t-1}|\theta)=\sum_{j=1}^{t-1}L(\mathcal{D}_{j}|\theta) is the likelihood function of previous t t response. Furthermore, the Maximum Posterior Weighted Information [ 62 , 63 ] further weights the Fisher and KL information with an additional posterior probability distribution P ( θ | 𝒟 1 : t − 1 ) P(\theta|\mathcal{D}_{1:t-1}) : ℐ j ( θ ^ t ) = ∫ θ ^ t − δ θ ^ t + δ P ( θ | 𝒟 1 : t − 1 ) L ( 𝒟 1 : t − 1 | θ ) ℐ j ( θ ) d θ \mathcal{I}_{j}(\hat{\theta}^{t})=\int_{\hat{\theta}^{t}-\delta}^{\hat{\theta}^{t}+\delta}P(\theta|\mathcal{D}_{1:t-1})L(\mathcal{D}_{1:t-1}|\theta)\mathcal{I}_{j}(\theta)d\theta , where ℐ ⁡ ( θ ) \mathcal{I}(\theta) can be the Fisher Information or KL divergence. Maximum Expected Information [ 62 ] accounts for all possible outcomes y t y_{t} and their impact on the updated proficiency estimate when weighting Fisher information. Lastly, the theta-Optimization (thOpt) process [ 64 ] selects questions by aligning the maximized information with the current proficiency estimate. Some non-parametric machine learning methods (e.g., decision trees) can also be explored, showing that a small number of questions can match or exceed traditional methods in accuracy, especially under high-dimensional and imbalanced data conditions [ 65 ] . 

 
 
 The selection algorithms discussed earlier are based on IRT and do not directly apply to other measurement models. As noted in Section IV , measurement models like CDM represent proficiency θ \theta as discrete states across different knowledge concepts. Adaptive selection algorithms under such models is known as Cognitive Diagnosis CAT (CD-CAT) [ 45 ] . While the selection principles remain similar (maximizing information from selected items), the information measures differ. In CD-CAT, techniques based on KL divergence [ 66 ] and Shannon entropy [ 67 ] are commonly used. Although continuous traits (as modeled in IRT) and discrete states (as modeled in DINA) describe different aspects of proficiency, they are complementary. This has led to the development of dual-objective CD-CAT methods [ 68 , 69 , 70 ] that aim to assess both simultaneously .

 
 
 Discussion: Selection algorithms in CAT have primarily relied on the above statistical heuristic approaches, which require domain experts to consider every possible testing scenario and manually design corresponding selection algorithms. These methods are model-specific , requiring distinct selection algorithms for different measurement models. For example, the above Fisher information [ 14 ] is specifically crafted for (M)IRT.
Consequently, previous statistical methods lack flexibility, and the selection algorithm must be re-designed if the underlying measurement model changes.

 
 
 Fig. 3: The Active Learning Framework and the relationship/correspondence between each component of Active Learning and those of CAT. 
 
 
 

### V-B Active Learning Algorithms 

 
 To design selection algorithms that are effective across different measurement models i.e., model-agnostic , researchers explore a general machine learning technique for data selection: Active Learning [ 71 ] . Active Learning is to actively choose some valuable data, thus can train better models with less data. This technique has improved data efficiency in numerous learning tasks [ 72 ] .

 
 
 As shown in Fig. 3 , active learning operates in cycles: where a selection algorithm iteratively chooses unlabeled samples based on the model’s current performance and queries a human annotator for labels. This process augments limited labeled data to improve model performance. The core challenge lies in designing effective sample selection algorithms, typically based on two criteria: informativeness , selecting samples that reduce model uncertainty [ 73 ] , and representativeness , selecting samples that reflect the overall data distribution [ 74 , 75 ] , or a combination of both [ 76 ] . Active learning shares a similar structure with CAT. Here, the measurement model plays the role of the learning model, the question selection corresponds to sample selection, and examinee responses serve as annotations. The goal is to estimate proficiency using as few questions as possible. This model-agnostic perspective avoids reliance on specific measurement model assumptions. Bi et al. [ 16 ] propose MAAT, a model-agnostic adaptive testing framework that evaluates the change (analogous to a gradient) in proficiency estimates after each response:

 

 
 | 
 j t + 1 = arg max q j ∈ 𝒬 𝔼 y j ‖ ∇ θ L ( 𝒟 1 : t ∪ { ( q j , y j ) } | θ ) ‖ . j_{t+1}=\arg\max_{q_{j}\in\mathcal{Q}}\mathbb{E}_{y_{j}}\left\|\nabla_{\theta}L(\mathcal{D}_{1:t}\cup\{(q_{j},y_{j})\}|\theta)\right\|. | 
 | 
 (9) | 
 

 Since the true responses to candidate questions are not available during selections, it computes the expected gradient norm with respect to the response label y y for each candidate question. This expectation quantifies the potential impact of each question on the proficiency estimation. The intuition behind this framework is that it prefers questions that are likely to most influence the proficiency estimation (i.e., have the greatest impact on its parameters) . Notably, this approach places no restriction on the specific type of measurement model, as long as it supports gradient-based optimization. 

 
 
 Discussion: 
With the rapid development of intelligent testing platforms (ranging from human’s online testing systems to AI model’s evaluation leaderboards), large-scale examinee response data has been accumulated. However, such data cannot be effectively leveraged by the above rules-based approaches (i.e., statistical algorithms and Active Learning algorithms) [ 26 , 25 ] . In contrast, recent data-driven approaches based on Reinforcement Learning (Section  V-C ) and Meta-Learning (Section  V-D ) have gained increasing attention. They automatically learn/optimize effective selection algorithms from large-scale response data without relying on manually defined heuristics or rules, and have demonstrated superior performance. 

 
 
 

### V-C Reinforcement Learning Algorithms 

 
 Reinforcement learning (RL), a subfield of machine learning, is a powerful approach that enables an agent to learn how to make optimal decisions automatically [ 77 ] . It has been successfully applied in various domains, including robotics, autonomous vehicles, education, and healthcare [ 78 , 79 ] . In RL, an agent interacts with an environment and receives feedback in the form of rewards or penalties based on its actions. The goal is to learn a policy π \pi , which can maximize the long-term cumulative reward. As shown in Fig.  4 , the policy can be learned by exploring the environment and learning from its consequences of actions. In essence, researchers in CAT utilize RL methodologies to address a question: Can the selection algorithm (policy) be automatically learned and optimized from data or examinee interactions, thus circumventing the necessity for expert intervention? 

 
 
 Markov Decision Process Formulation. The interaction between agent and environment can be viewed as a Markov Decision Process (MDP) [ 80 ] . Specifically, at each step, the agent observes current environment’s state ( s s ), and interacts with the environment by selecting its actions ( a a ). Simultaneously, the agent receives a reward ( r r ) from these interactions, influencing or changing the current state of the environment. The objective is to select a best sequence of actions, resulting in the highest cumulative reward ( ∑ t r t \sum_{t}r_{t} ). Therefore, most RL problems are formally described as estimating the optimality of the agent’s behavior in a given state (value-based methods [ 77 ] ) or the optimality of the action policy itself (policy-based methods [ 81 ] ) or the hybrid approaches [ 82 ] . The overall RL framework for CAT is illustrated in Fig.  4 . We formulate the CAT problem as an MDP, where the key RL components in the testing system are defined below:

 
 • 
 
 State : A state s t ∈ 𝒮 s_{t}\in\mathcal{S} represents the current condition or situation at each test step t t . It captures relevant information about examinee and the CAT system. Generally, the state includes the examinee’s previous response sequence (or a latent vector to represent the current proficiency estimate [ 83 ] ) and the candidate questions in the question bank [ 24 , 84 ] : s t = ( { q 1 , y 1 , … , q t , y t } , 𝒬 ) s_{t}=(\{q_{1},y_{1},...,q_{t},y_{t}\},\mathcal{Q}) 1 1 
 1 
 
 
 
 This aligns with the labeled data (answered questions), and the unlabeled data (question bank) in active learning (Section 5.2) .

 

 • 
 
 Action : An action a a refers to the choices that the CAT system can take in current state s t s_{t} , i.e., the selection of the next question from bank q t + 1 ∈ 𝒬 q_{t+1}\in\mathcal{Q} .

 

 • 
 
 Transition : The transition function is the probability of seeing state s t + 1 s_{t+1} after taking action q t q_{t} at current state s t s_{t} : P ⁡ ( s t + 1 | s t , q t + 1 ) P(s_{t+1}|s_{t},q_{t+1}) . At each step, the uncertainty comes from the examinee’s response correctness label y t + 1 y_{t+1} to question q t + 1 q_{t+1} .

 

 • 
 
 Reward : A reward r r is a scalar feedback that the CAT receives after selecting a question for the examinee. To achieve CAT’s goal in Definition 1 , the reward function can be defined as the accuracy of proficiency estimation at each step 2 2 
 2 
 
 
 
 As the true value is often unobtainable, it is commonly derived through simulation experiments. [ 85 , 84 , 86 ] , i.e., ‖ θ ^ t − θ 0 ‖ \|\hat{\theta}^{t}-\theta_{0}\| , or the performance prediction loss of θ ^ t \hat{\theta}^{t} on the held-out response data 𝒟 \mathcal{D} [ 24 , 87 ] , i.e., L ⁡ ( 𝒟 | θ ^ t ) L(\mathcal{D}|\hat{\theta}^{t}) . This reward signal is pivotal in guiding the policy π \pi to select the best-fitting question that can reduce the estimation error.

 

 
 
 
 Fig. 4: The overall Reinforcement Learning framework of CAT. The objective is to optimize the selection algorithm π \pi (i.e., policy) by exploring the large-scale examinee response data (i.e., environment). 
 
 
 Recently, with the advancements in deep learning, an increasing number of studies are leveraging Deep Reinforcement Learning to tackle the MDP problems in CAT. Li et al. [ 88 ] utilize the Deep Q-Network to represent the action-value function Q w ​ ( s , q ) Q_{w}(s,q) , representing the value of choosing question q q in state s s , and w w denotes its parameter of the network’s fully connected layer. The most suitable question is selected according to the policy:

 

 
 | 
 π ∗ ​ ( q | s ) = arg ⁡ max q ∈ 𝒬 ​ Q w ​ ( s , q ) . \pi^{*}(q|s)=\arg\max_{q\in\mathcal{Q}}{Q}_{w}(s,q). | 
 | 
 (10) | 
 

 To further capture the complex interactions between examinees and questions in practical testing scenarios, a Transformer-based Q-Network named NCAT [ 24 ] has been proposed. NCAT incorporates multiple functional modules, including a Double-Channel Performance Learning module that independently captures diverse aspects of examinee performance, and a Contradiction Learning module that identifies and extracts inconsistencies in examinee behavior, such as guessing and slipping. 

 
 
 Stochastic Shortest Path Formulation. Furthermore, CAT can be defined as a Stochastic Shortest Path (SSP) problem [ 89 ] , which is a special case of MDP. In an SSP, the objective is to find the shortest path (i.e., the minimum test step) from a given initial state s 0 s_{0} to goal states. In CAT, the goal state typically represents the completion of the test or the attainment of a predetermined level of proficiency estimation precision. Gilavert et al. [ 90 ] use Linear Programming to find the optimal testing policy π ∗ \pi^{*} , treating CAT like a flow network where each state must have balanced inflow and outflow (except for the start and end points). It denotes variables x s , a x_{s,a} as the expected accumulated occurrence frequency for every pair (state s ∈ 𝒮 s\in\mathcal{S} , qu q ∈ 𝒬 q\in\mathcal{Q} ), and equalizes i ​ n ​ ( s ) in(s) and o ​ u ​ t ​ ( s ) out(s) flow model for every state s s . The flow into a state s s is the sum of the expected frequencies of all actions in all other states s ′ s^{\prime} that lead to s s : i ​ n ​ ( s ) = ∑ s ′ , q x s ′ , q ​ P ​ ( s | s ′ , q ) {in}(s)=\sum_{s^{\prime},q}x_{s^{\prime},q}P(s|s^{\prime},q) . The flow out of a state s s is the sum of the expected frequencies of all actions in state s s : o ​ u ​ t ​ ( s ) = ∑ q x s , q {out}(s)=\sum_{q}x_{s,q} . The objective function is to maximize the total expected reward r r , which is the sum of the expected frequencies times the immediate rewards r ⁡ ( s , q ) r(s,q) for all state-action pairs: min ⁡ ∑ s ∈ 𝒮 , q ∈ 𝒬 x s , q ⁡ x s , q ​ r ​ ( s , q ) \min_{x_{s,q}}\sum_{s\in\mathcal{S},q\in\mathcal{Q}}{x_{s,q}r(s,q)} . Thus the optimal question selection policy π ∗ \pi^{*} can be obtained by:

 

 
 | 
 π ∗ ​ ( q | s ) = x s , q ∑ q ′ ∈ 𝒬 x s , q ′ . \pi^{*}(q|s)=\frac{x_{s,q}}{\sum_{q^{\prime}\in\mathcal{Q}}x_{s,q^{\prime}}}. | 
 | 
 (11) | 
 

 
 
 Partial-Observable MDP Formulation. Partial-Observable MDP (POMDP) extends the standard MDP framework to settings where the environment is only partially observable [ 91 ] . Traditional CAT models often assume that an examinee’s proficiency can be fully inferred from previous responses, allowing it to be treated as an MDP with the proficiency estimate as the state. However, in practice, proficiency cannot be perfectly inferred due to some inherent uncertainty [ 92 , 93 ] .

 
 
 To this end, many works [ 84 , 90 , 94 ] model CAT as a POMDP. Compared with MDP, the POMDP model has two additional elements. O O : A set of observations; Z Z : Observation probabilities. Z ⁡ ( o | s ′ , q ) Z(o|s^{\prime},q) is the probability of making observation o o after selecting question q q and transitioning to state s ′ s^{\prime} . While the underlying state (proficiency) remains, it is not fully observable. Instead, these methods maintains a belief state b ⁡ ( s ) b(s) , a probability distribution over possible proficiencies, which is updated via Bayes’ rule: When the agent select action (question) q q in belief state b b and makes observation o o , it updates its belief state to b ′ ​ ( s ′ ) b^{\prime}(s^{\prime}) : b ′ ​ ( s ′ ) = η ​ Z ​ ( o | s ′ , q ) ​ ∑ s ∈ 𝒮 P ⁡ ( s ′ | s , q ) ​ b ​ ( s ) b^{\prime}(s^{\prime})=\eta{Z(o|s^{\prime},q)\sum_{s\in\mathcal{S}}P(s^{\prime}|s,q)b(s)} . where η \eta is a normalizing constant. POMDPs can be solved by many algorithms, such as Grid-based algorithms [ 95 ] , Monte Carlo tree search [ 96 ] . However, due to its partial observability of the environment, POMDPs are more challenging to solve than MDPs.

 
 
 

### V-D Meta Learning Algorithms 

 
 Another data-driven machine learning approach that can address this complex CAT problem is meta-learning [ 97 ] : It involves training a model on various tasks to acquire cross-task knowledge or learn how to learn efficiently. Specifically, the base-learner is trained across a variety of related tasks, allowing it to gather cross-task insights and general knowledge about how to learn efficiently. Then, the meta-learner leverages this knowledge to swiftly adapt to new, unseen tasks [ 98 ] . In CAT, each examinee’s testing process can be seen as a task because it involves selecting appropriate test questions based on the proficiency level. The selection algorithm can be regarded as a form of general knowledge because it represents the accumulated knowledge and experience gained from a diverse set of examinees (Fig. 5 ). This knowledge can include the best policy for question selection, information about the characteristics of different test questions, the examinee proficiency prior, etc. By learning from these diverse examinees (tasks) in the large-scale response dataset, it can acquire a good question selection that can adapt to individual examinees.

 
 
 Fig. 5: The overall Meta Learning framework of CAT, and this figure is adapted from [ 26 ] . The objective is to optimize the selection algorithm π \pi by exploring the large-scale examinee response data. 
 
 
 Bi-Level Optimization. Bi-Level optimization is a classical meta-learning approach commonly applied in CAT. It decomposes the learning process into two nested levels: an inner level that adapts to individual examinees and an outer level that learn general knowledge. Ghosh et al. [ 26 ] propose a bi-level optimization framework for CAT (BOBCAT) to directly learn the data-driven selection algorithm π \pi . Specifically: let N N denote the number of examinees in the response dataset for training π \pi . The responses of each examinee i i are randomly divided into a support set 𝒟 s i \mathcal{D}_{s}^{i} and a query set 𝒟 u i \mathcal{D}_{u}^{i} , where π \pi sequentially select a total of t t questions { q 1 , … , q t } \{q_{1},...,q_{t}\} from 𝒟 s i \mathcal{D}_{s}^{i} , observe their responses, and predict their response on the held-out query set 𝒟 u i \mathcal{D}_{u}^{i} . The global knowledge (i.e., selection algorithm π \pi and global parameters γ \gamma ) is redefined as the objective of bi-level optimization:

 

 
 | 
 | 
 min π , γ ⁡ 1 N ​ ∑ i = 1 N ∑ ( q , y ) ∈ 𝒟 u i ℓ ⁡ ( y , f ⁡ ( q , θ ^ i ) ) , \displaystyle\min_{\pi,\gamma}\frac{1}{N}\sum_{i=1}^{N}{\sum_{(q,y)\in\mathcal{D}_{u}^{i}}{\ell(y,f(q,\hat{\theta}_{i}))}}, | 
 | 
 (12) | 
 
 
 | 
 | 
 s . t . θ ^ i = arg ⁡ min θ i ⁡ ∑ ( q , y ) ∈ 𝒟 s i ℓ ⁡ ( y , f ⁡ ( q , θ i ) ) , \displaystyle\mathrm{s.t.}\;\;\hat{\theta}_{i}=\mathop{\arg\min}_{\theta_{i}}{\sum_{(q,y)\in\mathcal{D}_{s}^{i}}{\ell\left(y,f\left(q,\theta_{i}\right)\right)}}, | 
 | 
 (13) | 
 
 
 | 
 | 
 where q t + 1 ∼ π ⁡ ( q | q 1 , y i ⁡ ( 1 ) , … , q t , y i ⁡ ( t ) ) ∈ 𝒟 s i . \displaystyle\mathrm{where}\quad q_{t+1}\sim\pi\left(q|q_{1},y_{i(1)},...,q_{t},y_{i(t)}\right)\in\mathcal{D}_{s}^{i}. | 
 | 
 (14) | 
 

 Fig. 5 shows the overall meta learning framework. In the inner-level (Eq.( 13 )), the question in the support set 𝒟 s i \mathcal{D}_{s}^{i} for examinee i i is sequentially selected by π \pi , according to the previous responses; then binary cross-entropy loss ℓ ⁡ ( ⋅ ) \ell(\cdot) on 𝒟 s i \mathcal{D}_{s}^{i} is minimized for estimating the proficiency θ ^ i \hat{\theta}_{i} for the outer-level. In the outer-level (Eq.( 12 )), the loss of the estimate θ ^ i \hat{\theta}_{i} on the query set 𝒟 u i \mathcal{D}_{u}^{i} is minimized to learn the selection algorithm π \pi and the global parameters γ \gamma (e.g., question characteristics). The algorithm π \pi is also model-agnostic. It could be adapted to the given measurement model ( f f ) automatically by optimizing this problem for efficient selection.

 
 
 Through large-scale sampling and training, this framework learns to estimate and quantify the value of each question for different examinees and under varying contexts. Even for questions whose IDs do not appear in the training set, their value can be inferred from their characteristics via γ \gamma . Once the question selection algorithm is trained, its parameters do not update during the CAT process and adaptively select the next question based on previous response behaviors.

 
 
 Based on BOBCAT, there have been increasing efforts to improve upon it. Ma et al. [ 85 ] propose a flexible optimization framework Decoupled Learning CAT (DL-CAT). The original BOBCAT obtains the parameters of two modules (i.e., examinee proficiency estimation and question selection algorithm) through coupled inner and outer optimizations, i.e., the result of the outer optimization model is used to measure the quality of the inner. DL-CAT devises a ground-truth construction strategy, and a pairwise loss function, allowing these two models to be trained independently; Feng et al. [ 99 ] introduces a constrained version of BOBCAT to address the question exposure and test overlap issues. Yu et al. [ 100 ] recently introduce the collaborative information of examinees in optimizing this bi-level problem, achieving fast convergence of proficiency estimation.

 
 
 Meta-Learning vs Reinforcement Learning. In CAT, meta-learning methods can be seen as a higher-level learning process that learns how to adapt a general strategy for question selection to specific examinees based on their responses. Actually, it can be reframed as an RL problem. Zhuang et al. [ 24 ] propose NCAT to transform the meta-learning problem in CAT into an RL problem. Because the test may stop at any step according to different stopping rules, NCAT simplifies the original objective (Eq( 12 )) and sums all the steps to minimize the loss:

 

 
 | 
 | 
 | 
 min π ⁡ 1 N ​ ∑ i = 1 N ∑ t = 1 T ∑ ( q , y ) ∈ 𝒟 u i ℓ ⁡ ( y , f ⁡ ( q , θ ^ i t ) ) \displaystyle\min_{\pi}\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}{\sum_{(q,y)\in\mathcal{D}_{u}^{i}}{\ell(y,f(q,\hat{\theta}_{i}^{t}))}} | 
 | 
 (15) | 

 
 | 
 | 
 ≜ \displaystyle\triangleq | 
 max π 𝔼 i ∼ π [ ∑ t = 1 T − ∑ ( q , y ) ∈ 𝒟 u i ℓ ( y , f ( q , θ ^ i t ) ) ] \displaystyle\mathop{\max}\limits_{\pi}\mathbb{E}_{i\sim\pi}\left[\sum_{t=1}^{T}{-{\sum_{(q,y)\in\mathcal{D}_{u}^{i}}{\ell(y,f(q,\hat{\theta}_{i}^{t}))}}}\right] | 
 | 

 
 | 
 | 
 = \displaystyle= | 
 max π 𝔼 i ∼ π [ ∑ t = 1 T − L ( 𝒟 u i | θ ^ i t ) ] , \displaystyle\max_{\pi}\mathbb{E}_{i\sim\pi}\left[\sum_{t=1}^{T}{-L(\mathcal{D}_{u}^{i}|{\hat{\theta}_{i}^{t}})}\right], | 
 | 
 

 where θ ^ i t = arg ⁡ min θ i ⁡ ∑ ( q , y ) ∈ 𝒟 s i ⁡ ( t ) ℓ ⁡ ( y , f ⁡ ( q , θ i ) ) \hat{\theta}_{i}^{t}=\mathop{\arg\min}_{\theta_{i}}{\sum_{(q,y)\in\mathcal{D}_{s}^{i(t)}}{\ell\left(y,f\left(q,\theta_{i}\right)\right)}} and 𝒟 s i ⁡ ( t ) = { q 1 , y i ⁡ ( 1 ) , … , q t , y i ⁡ ( t ) } \mathcal{D}_{s}^{i(t)}=\{q_{1},y_{i(1)},...,q_{t},y_{i(t)}\} . Thus, the bi-level optimization is transformed into maximizing the expected
cumulative reward (i.e., − L ⁡ ( 𝒟 u i | θ ^ i t ) -L(\mathcal{D}_{u}^{i}|{\hat{\theta}_{i}^{t}}) ) in RL settings, where the reward is the negative loss of the estimated proficiency of examinee i i on the query set at step t t . Recently, GMOCAT [ 86 ] has been proposed as a Multi-Objective RL framework. GMOCAT uses Graph Neural Networks to capture the complex relationships between questions and skills. It adopts an Actor-Critic architecture and incorporates three objectives into the reward function: (1) improving prediction accuracy, (2) enhancing concept (skill) diversity, and (3) reducing question exposure. 

 
 
 Discussion: The aforementioned data-driven machine learning approaches, i.e., Reinforcement Learning and Meta Learning, are capable of uncovering latent patterns and correlations from data, and directly optimizing question selection policies. By fitting to large-scale data, they can approximate the ultimate goal of CAT. However, potential issues such as data bias, model overfitting, and high training overhead should not be overlooked. 

 
 
 Fig. 6: Illustration of the subset optimization problem, adapted from [ 21 ] : Selecting subset S S to cover the bank Q Q . Rectangles represent different questions, with w ⁡ ( i , j ) w(i,j) measuring the similarity of question pair. 
 
 
 

### V-E Subset Selection Algorithms 

 
 The ultimate objective of CAT is to measure examinees’ abilities both efficiently and accurately. Specifically, as illustrated in Definition 1 , the goal is to find a subset S S of T T questions from question bank 𝒬 \mathcal{Q} , so that the final proficiency estimate θ ^ T \hat{\theta}^{T} can approach the true proficiency θ 0 \theta_{0} :

 

 
 | 
 min | S | = T ‖ θ ^ T − θ 0 ‖ , \mathop{\mathrm{min}}\limits_{|S|=T}\|{\hat{\theta}^{T}}-\theta_{0}\|, | 
 | 
 (16) | 
 

 where θ ^ T = arg ⁡ min ⁡ ∑ ( q , y ) ∈ S θ ⁡ ℓ ⁡ ( y , f ⁡ ( q , θ ) ) \hat{\theta}^{T}=\arg\min_{\theta}{\sum_{(q,y)\in S}{\ell(y,f(q,\theta))}} is the final proficiency estimate when the test ends with the corresponding T T responses. In contrast to previous sequential selection methods, it essentially doesn’t require perfect selection at each step, but rather emphasizes the accuracy of the final estimate. 

 
 
 From a global perspective, CAT essentially is a Subset Selection problem [ 101 ] , a fundamental challenge in machine learning and optimization. It revolves around choosing a subset of elements S S from a larger set 𝒬 \mathcal{Q} that optimizes a particular objective function F ⁡ ( S ) F(S) while adhering to specific constraints. However, we cannot directly solve the above optimization problem due to the following main challenge: The true proficiency of the examinee, denoted by θ 0 \theta_{0} , is unknown. It is not available in the dataset, which prevents us from directly optimizing or designing the question selection algorithm. To address this issue, some researchers have developed heuristic methods. For example, Mujtaba et al. [ 102 ] use the standard error of measurement as the objective F ⁡ ( S ) F(S) , which provides a measure of confidence in an estimate from a test. At each step, it uses multi-objective evolutionary algorithms to obtain the set of Pareto-optimal solutions [ 103 ] by maximizing precision and minimizing the number of questions. Recently, for AI model evaluation, clustering techniques (e.g., K-means) have been used to select representative subsets S S from benchmarks [ 38 ] .

 
 
 To develop a more general and scalable CAT framework, Zhuang et al. [ 21 ] propose BECAT, which reformulates the question selection problem in a data summary manner. Since the true proficiency θ 0 \theta_{0} is unobservable, they approximate it using θ ∗ \theta^{*} : the proficiency estimated from an examinee’s full responses to the entire question bank 𝒬 \mathcal{Q} , i.e., θ ∗ ≈ θ 0 \theta^{*}\approx\theta_{0} . This approximation enables the selection algorithm to target θ ∗ \theta^{*} instead of the unknown θ 0 \theta_{0} : Select a subset of questions S ⊆ 𝒬 S\subseteq\mathcal{Q} such that the estimated proficiency based on S S closely approximates θ ∗ \theta^{*} (i.e., the estimate that would be obtained if optimizing on the full responses to 𝒬 \mathcal{Q} ).

 

 
 | 
 | 
 min | S | = T ⁡ ‖ θ ^ T − θ 0 ‖ ⇒ min | S | = T ⁡ ‖ θ ^ T − θ ∗ ‖ \displaystyle\min_{|S|=T}\|{\hat{\theta}^{T}}-\theta_{0}\|\Rightarrow\min_{|S|=T}\|{\hat{\theta}^{T}}-\theta^{*}\| | 
 | 
 
 
 | 
 ⇒ \displaystyle\Rightarrow | 
 min | S | = T max θ ∈ Θ ∥ ∑ ( q , y ) ∈ S γ ∇ ℓ ( y , f ( q , θ ) ) − ∑ ( q , y ) ∈ 𝒬 ∇ ℓ ( y , f ( q , θ ) ) ∥ \displaystyle\min_{|S|=T}\max_{\theta\in\Theta}\Big\|\sum_{(q,y)\in S}{\gamma\nabla\ell(y,f(q,\theta))}-\sum_{(q,y)\in\mathcal{Q}}{\nabla\ell(y,f(q,\theta))}\Big\| | 
 | 
 
 
 | 
 ⇒ \displaystyle\Rightarrow | 
 min | S | = T max θ ∈ Θ ∑ i ∈ 𝒬 min j ∈ S ​ ‖ ∇ ℓ i ​ ( θ ) − ∇ ℓ j ​ ( θ ) ‖ \displaystyle\mathop{\mathrm{min}}\limits_{|S|=T}\mathop{\mathrm{max}}\limits_{\theta\in\Theta}\sum_{i\in\mathcal{Q}}\mathrm{min}_{j\in S}\|\nabla\ell_{i}(\theta)-\nabla\ell_{j}(\theta)\| | 
 | 
 
 
 | 
 ⇒ \displaystyle\Rightarrow | 
 max | S | = T ∑ i ∈ 𝒬 max j ∈ S ​ w ​ ( i , j ) , \displaystyle\mathop{\mathrm{max}}\limits_{|S|=T}\sum_{i\in\mathcal{Q}}\mathrm{max}_{j\in S}\;w(i,j), | 
 | 
 (17) | 
 

 where w ⁡ ( i , j ) ≜ d − max θ ∈ Θ ​ ‖ ∇ ℓ i ​ ( θ ) − ∇ ℓ j ​ ( θ ) ‖ w(i,j)\triangleq d-\mathrm{max}_{\theta\in\Theta}{\|\nabla\ell_{i}(\theta)-\nabla\ell_{j}(\theta)\|} is the gradient similarity between question pair ( q i , q j ) (q_{i},q_{j}) for this examinee, thus the objective function F ⁡ ( S ) = ∑ i ∈ 𝒬 max j ∈ S ​ w ​ ( i , j ) F(S)=\sum_{i\in\mathcal{Q}}\mathrm{max}_{j\in S}\;w(i,j) . The core of BECAT’s subset selection algorithm is to find a subset S S of size T T that maximizes the coverage of 𝒬 \mathcal{Q} , quantified by the similarity measure w ⁡ ( i , j ) w(i,j) . This approach (Fig.  6 ) essentially seeks the most representative questions, aligning with prior selection algorithms but under a new, more rigorous theoretical framework.

 
 
 Given the NP-Hard nature of this optimization, BECAT employs a submodular function approximation. A simple greedy algorithm can finally solve this subset selection problem, with BECAT ensuring that the estimate error remains upper-bounded at each step. The subset selection problem in CAT is a fresh direction with significant potential. This method offers a universal framework for question selection, applicable across various complex measurement models that can utilize gradient-based estimations, including neural network models.

 
 
 TABLE I: Comparison of Different Question Selection Algorithms in CAT 
 
 
 
 
 
 Category 
 | 
 
 
 Generality 
 | 
 
 
 Interpretability 
 | 
 
 
 Need Training 
 | 
 
 
 Advantages 
 | 
 
 
 Disadvantages 
 | 

 
 
 
 Statistical    Algorithms 
 | 
 
 
 ✗ 
 | 
 
 
 ✓ 
 | 
 
 
 ✗ 
 | 
 
 
 Simple implementation and efficient operation 
 | 
 
 
 Dependent on IRTs and requires expert knowledge for design 
 | 

 
 
 
 Active Learning 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✗ 
 | 
 
 
 Model-agnostic and flexible 
 | 
 
 
 Neglect the nuanced information within measurement model parameters 
 | 

 
 
 
 Reinforcement Learning 
 | 
 
 
 ✓ 
 | 
 
 
 ✗ 
 | 
 
 
 ✓ 
 | 
 
 
 Automatic generation of selection algorithm; Sequential Decision Making 
 | 
 
 
 Incurs additional training costs and potential bias from data-driven selection 
 | 

 
 
 
 Meta Learning 
 | 
 
 
 ✓ 
 | 
 
 
 ✗ 
 | 
 
 
 ✓ 
 | 
 
 
 Automatic generation of selection algorithm; Fast Adaptation 
 | 
 
 
 Incurs additional training costs and potential bias from data-driven selection 
 | 

 
 
 
 Subset Selection 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✗ 
 | 
 
 
 Strong theoretical guarantees for estimation accuracy 
 | 
 
 
 Faces challenges in the initial stages of CAT 
 | 

 
 
 
 Discussion: It is noteworthy that, despite the superior performance demonstrated by the latest machine learning and deep learning approaches [ 104 ] , they have not yet replaced traditional statistical approaches in practice . Particularly in testing scenarios that prioritize interpretability or efficiency, statistical methods remain predominant . In the Appendix, we compare these five categories of selection algorithms in CAT systems, highlighting the generality and interpretability of each category, along with their main advantages and limitations. This overview assists researchers in identifying the most suitable algorithm for their CAT applications, balancing efficiency and complexity.

 
 
 
 

## VI Question Bank Construction 

 
 To develop a high-quality CAT, the foundational step is to construct a high-quality question bank. The bank construction can be decomposed into two main stages: Question Characteristics Analysis and Question Bank Development: (1) Question Characteristics Analysis first detailedly examines the properties and attributes of potential questions. Then, (2) Question Bank Development assembles the final question bank 𝒬 \mathcal{Q} from the analyzed questions.

 
 

### VI-A Question Characteristics Analysis 

 
 The first stage, question characteristics analysis, involves a detailed examination of the properties and attributes of potential questions, e.g., difficulty, discrimination, and the knowledge concepts required to answer the question. For example, when selecting questions based on Fisher Information, one must leverage pre-calibrated parameters like difficulty ( β j \beta_{j} ), discrimination ( α j \alpha_{j} ), and guessing factor ( c j c_{j} ), alongside the current proficiency estimate, to compute the Information value ℐ j ​ ( θ ) \mathcal{I}_{j}(\theta) for each question j j . The methods of characteristics analysis can be categorized into three main approaches: expert-based, statistic-based, and deep learning-based methods.

 
 
 Expert-based Characteristics Annotation. In expert-based annotation, domain experts assess question parameters, as seen in online CAT systems like SIETTE [ 105 ] and GenTAI [ 106 ] . Effective expert estimation often involves structured questionnaires [ 107 ] , followed by discussions to resolve divergent opinions. Results are aggregated using averages for continuous attributes or voting for discrete ones [ 108 ] . Expert judgments can be subjective, leading to potential inaccuracies, especially with limited or inconsistent expert input. With the advancement of generative AI, LLMs can also be used to annotate question characteristics [ 109 ] .

 
 
 Statistic-based Characteristics Annotation. The statistic-based method for annotating question characteristics requires gathering responses from a large group of examinees. It is resource-intensive nature and involves pre-testing with examinees [ 110 ] . In Classic Test Theory, question difficulty is calculated as the proportion of correct responses within examinees [ 111 , 112 ] , while discrimination is derived from performance disparities between higher and lower ability examinees [ 113 ] . The Q-matrix is another crucial characteristic of questions. It is a binary matrix that indicates which knowledge concepts are associated with a question. Numerous researchers have attempted to employ some parameter estimation approaches (e.g., maximum likelihood estimation and Bayesian estimation), to learn these characteristic parameters from response data [ 114 , 115 , 116 ] .

 
 
 Deep Learning-based Characteristics Annotation. With the rise of Natural Language Processing (NLP), there has been an increasing trend in recent years to directly use the textual information of questions to analyze various attributes. For difficulty prediction, attention-based CNN models and domain adaptation strategies have been used to evaluate reading questions and medical question complexity [ 117 , 23 , 118 ] . For knowledge concept (Q-matrix) prediction, which typically exhibits a hierarchical structure, a Hierarchical attention-based Recurrent Neural Network has been proposed [ 119 , 120 ] . Lei et al. [ 121 ] further take into account the multi-modal features of questions, such as images and formulas. Pre-trained NLP models have also proven effective for automated question analysis [ 122 , 123 ] .

 
 
 

### VI-B Question Bank Development 

 
 The second stage, question bank development, involves the actual assembly of the question bank from the analyzed questions of the first stage. This process should aim to create a balanced and varied bank that can cater to different levels of proficiency and different areas of knowledge [ 124 , 125 , 126 ] . According to different scenarios, the approaches to developing a question bank can be categorized into the following three aspects.

 
 
 Question Bank Blueprint Design. The goal of the blueprint design is to create an optimal framework for a bank, outlining the distribution of questions based on various attributes. Reckase et al. [ 125 ] analyze the characteristics of an optimal question bank in a CAT system using a 1PL-IRT model with a maximum Fisher information selection algorithm. They propose the bin-and-union method to allow a maximum deviation r r between optimal difficulty and estimated proficiency, extending these methods for large-scale CAT systems and continuous new question pretesting [ 127 , 128 ] .

 
 
 Question Bank Assembly. While the blueprint design focuses on creating an optimal framework, the assembly process involves generating question banks from an existing master bank according to specific requirements. Way et al. [ 129 ] discussed the development and maintenance of a master bank, including constraints to ensure the assembled question bank meets desired specifications. A mixed-integer programming [ 130 ] was proposed to create a bank that satisfies content specifications and maximizes information at selected proficiency values.

 
 
 Question Bank Rotating. Rotating the question bank involves dividing a master bank into smaller banks with overlapping elements, ensuring balanced exposure rates [ 131 , 132 , 133 ] . Ariel et al. [ 132 ] proposed dividing a master bank into smaller banks using Gulliksen’s matched random subtests method [ 134 ] to prevent over- or underexposure. The Weighted Deviation Model [ 133 ] manages the degree of overlap, maintaining representativeness and preventing question overexposure.

 
 
 Discussion: 
Think of the entire bank development process as creating and managing a library. The blueprint design is like the architectural plan for the library, defining where each section (e.g., fiction, non-fiction) will be located; The assembly process resembles acquiring books from suppliers based on specific demands; Rotating the question bank is similar to periodically rotating the books on display. Even though the library has a vast collection, only a subset is displayed prominently at any given time. This rotation ensures that different books get exposure, and library visitors encounter a variety of books over time. Although this section has so far focused on classical methods, the bank construction pipeline can also incorporate LLMs as auxiliary components. In other words, LLMs can be integrated into the bank development stage to improve scalability and reduce manual cost, while the psychometric principles of CAT remain unchanged. 
 In this analogy, LLMs (or agents) can be viewed as “librarians” that help scale and accelerate curation: they can draft candidate items on demand under explicit constraints, and generate useful metadata (topic tags, expected solution outlines, and common error patterns) that supports indexing and retrieval. 
The construction of a high-quality question bank introduced in this section is not just a prerequisite for CAT, but also a continuous process. It requires regular updates and refinements to ensure the relevance and effectiveness of the adaptive testing system.

 
 
 
 

## VII Test Control of CAT 

 
 When implementing a testing system, in addition to considering the three components mentioned above, several key factors need to be taken into consideration, such as exposure control, fairness, robustness, and search efficiency.

 
 

### VII-A Exposure Control 

 
 Exposure control aim to balance the frequency of each question’s use from the question bank . Proper exposure control can help mitigate the risk of overexposure of questions , minimize question waste, and maximize test coverage. Two popular strategies for exposure control are the Sympson-Hetter method [ 135 ] and the A-Stratified method [ 136 ] : (1) Sympson-Hetter Method manages question exposure rates using conditional probabilities. It doesn’t assign a selected question to the examinee immediately; instead, it passes through a probability filter. The actual chance a question is given to an examinee depends on both its selection likelihood and a exposure control parameter , keeping question exposure within acceptable limits. However, this method may not effectively increase the usage rate of low-exposure questions. Enhancements to this method have been developed to address these limitations [ 137 , 138 , 139 ] ; (2) A-Stratified Method and its subsequent researches [ 140 , 64 , 141 ] are designed to counteract selection biases of algorithms favoring certain questions (e.g., Fisher Information prefers highly differentiated questions). On the other hand, numerous studies [ 16 , 86 , 142 ] have attempted to incorporate the coverage of knowledge concepts as a criterion in question selection, aiming to make the assessment more comprehensive.

 
 
 This factor is crucial for both humans and AI. When students are familiar with exam questions beforehand, the test results lose credibility. Similarly, for AI model evaluations, it has been observed that benchmarks released before the creation date of LLM’s training data generally perform better than those released afterward [ 143 ] . Increasingly, the AI evaluation is being questioned regarding data contamination [ 144 ] . Therefore, controlling question exposure rates is a necessary measure to improve the reliability of assessments.

 
 
 

### VII-B Fairness 

 
 Fairness is a topic of profound societal significance in both education and machine learning research fields, sparking numerous discussions and leading to the development of many fairness-aware learning algorithms [ 145 , 146 , 147 ] . As a technology with potential applications in high-stakes testing, fairness in CAT is a paramount concern. In CAT, the bias that leads to fairness issues can be introduced through three components:

 
 
 
 • 
 
 Bias in Measurement Models. Biases in measurement models may stem from the skewed training data, which could reflect the underrepresentation of certain groups or pre-existing educational disparities [ 148 , 149 , 150 ] . Such biases can lead to an inaccurate and biased estimation of an examinee’s proficiency θ \theta , resulting in unfair outcomes.
 A practical mitigation is to evaluate calibration and model fit across subpopulations (e.g., invariance checks) and apply multi-group calibration when needed. Fairness-aware calibration objectives can also be used as a light regularizer to reduce spurious group effects. 

 

 • 
 
 Bias in Question Bank. The question bank may contain biases if questions are not equally applicable or relatable to all examinees, potentially disadvantaging certain groups [ 151 , 152 ] . For example, some questions in NAPLAN have been deemed unfair for rural examinees, as these questions don’t relate to their real-life experiences [ 153 ] . Various methods have been proposed to detect this type of bias [ 154 , 151 , 155 ] .
 For example, mitigation often follows an audit–repair loop: DIF analyses flag potentially biased items, which are then revised, replaced, or retired. This is usually paired with expert review to separate unintended context bias from construct-relevant differences. 

 

 • 
 
 Bias in Selection Algorithms. Selection algorithm can introduce bias since every algorithm has its own “selection preferences”. For example, the Maximum Fisher Information tends to select questions with high discrimination [ 14 ] . If such questions unexpectedly correlate with specialized knowledge known only to a specific group, bias may ensue.

 

 
 
 
 Concerns about fairness in CAT also stem from the fact that examinees answer different questions [ 156 ] . Equating, a technique used to ensure score equivalence across different tests, is commonly employed to address such concerns [ 157 ] . Many further studies about equating scores have been conducted [ 158 , 159 , 160 ] . In real-world tests such as the GRE, equating has been used to standardize scores and percentiles, taking into account the difficulty of the questions answered. This process ensures that scores can be compared fairly across different examinees worldwide.
 In practice, equating is complemented by routine drift checks and periodic DIF re-audits, especially when new items are added or rotated. This helps preserve comparability as the bank evolves. 

 
 
 

### VII-C Robustness 

 
 Noise in CAT can impact the precision of the estimated proficiency of an examinee, leading to potential errors in score interpretation. In CAT, noise usually refers to the random variability or measurement error that can affect the accuracy of estimation. It can arise from various sources such as test administration conditions, examinee behavior, or question characteristics. For example, an examinee may be distracted by environmental noise during the test, leading to an incorrect response that does not reflect their true ability. Alternatively, a poorly worded or ambiguous question may confuse examinees, introducing unintended variability in responses. 

 
 
 To mitigate the effects of noise in CAT, a robustness factor is introduced to help stabilize the estimation of proficiency by incorporating additional information, thereby counteracting the impact of noise and improving the reliability [ 161 ] . In machine learning, various robustness techniques are employed to enhance the performance of models in the presence of noise, such as regularization methods [ 162 ] , data augmentation [ 163 ] , adversarial methods [ 164 ] , ensemble methods [ 165 ] . In the CAT testing process, significant sources of noise such as guess and slip factors made by examinees, introduce uncertainty. For example, an examinee’s proficiency level may not be uniquely determined by their responses, as they may solve a particular question correctly using different knowledge concepts or even by guessing. The presence of noise and uncertainty poses a significant challenge to the robustness of CAT systems. Veldkamp et al. [ 166 ] consider the uncertainty in question parameters during the selection process. More recently, ensemble learning has been explored to combine multiple potential estimates at each step, thereby enhancing proficiency estimation [ 92 ] .

 
 
 

### VII-D Search Efficiency 

 
 In large-scale educational testing, efficient question selection is a critical challenge. Traditional selection algorithms often evaluate all candidate questions in a brute-force manner, resulting in a linear time complexity of O ⁡ ( | 𝒬 | ) O(|\mathcal{Q}|) , where 𝒬 \mathcal{Q} is the question bank. This becomes a computational bottleneck in intelligent testing systems. To mitigate this, some organizations like GMAT [ 167 ] rely on manual filtering rules crafted by experts, which is labor-intensive and lacks scalability. Recent research has explored two main directions to improve efficiency:

 
 • 
 
 Heuristic Search via PSO: Particle Swarm Optimization (PSO) has been applied in IRT-based adaptive testing [ 168 , 169 ] . PSO enables parallel exploration of the search space, where each particle represents a candidate question. This parallelism accelerates convergence toward optimal selections, reducing computational burden.

 

 • 
 
 Tree-Based Indexing: Inspired by recommendation systems and information retrieval, efficient search structures such as balanced trees have been adopted [ 170 , 171 ] . Hong et al. [ 172 ] propose a Search-Efficient CAT framework that employs examinee-aware space partitioning to construct a tree-based index. This method significantly narrows the search space and avoids redundant computations across testing rounds, reducing the search complexity from O ⁡ ( | 𝒬 | ) O(|\mathcal{Q}|) to O ⁡ ( log ⁡ | 𝒬 | ) O(\log|\mathcal{Q}|) .

 

 
 
 
 Discussion: 
While accuracy and efficiency stand as primary objectives, these factors hold significant importance for practical settings, especially in high-stakes testing scenarios (e.g., competitive or selective examinations). However, consideration of these factors may inevitably reduce accuracy. For example, when considering the additional fairness to ensure equity among different groups, it might be necessary to deviate from the optimal trajectory of a well-trained selection algorithm. Thus, CAT poses a multidimensional decision-making challenge, necessitating the consideration of various factors at the same time using diverse machine learning techniques. In the Appendix, we show the underlying causes and advantages of different factors in CAT test control.

 
 
 
 

## VIII Evaluation 

 
 Various metrics have been developed to assess the performance of CAT methods, such as correlation coefficients, bias, and measurement error [ 173 , 174 ] . This section introduces two of the most extensively utilized evaluation methods: simulation of proficiency estimation and examinee score prediction.

 
 
 Simulation of Proficiency Estimation. The simulation of ability estimation is a foundational evaluation technique in CAT [ 4 ] . Since true proficiency ( θ 0 \theta_{0} ) is unobservable, we simulate it by sampling a set of values { θ 0 1 , θ 0 2 , … , θ 0 N } \{\theta_{0}^{1},\theta_{0}^{2},...,\theta_{0}^{N}\} to represent a virtual group of examinees. This approach enables us to further emulate the interactions between examinees (with these proficiencies) and any question from the question bank, utilizing measurement models. Consequently, the estimated final proficiency values θ ^ T \hat{\theta}^{T} can be directly compared with the true values θ 0 \theta_{0} . For example, by computing the Mean Square Error (MSE), i.e., 𝔼 ​ ‖ θ ^ T − θ 0 ‖ \mathbb{E}\|\hat{\theta}^{T}-\theta_{0}\| , to evaluate the accuracy of the CAT system [ 16 , 45 ] .

 
 
 Examinee Score Prediction. In machine learning–based CAT systems, proficiency estimates are often validated by predicting whether examinees will answer unseen questions correctly. Typically, examinees are split into training, validation, and test sets (e.g., 70%-20%-10%), ensuring no overlap. The training set is used to calibrate item parameters (Section  VI-A ) and train selection algorithms (Sections  V-C , V-D ). During validation or testing, the responses of each examinee are further divided into a candidate set 𝒬 i \mathcal{Q}_{i} for selecting questions and a held-out meta set ℳ i \mathcal{M}_{i} for evaluation. The candidate set 𝒬 i \mathcal{Q}_{i} (with corresponding response label y y ) is used to simulate the CAT procedure: Selecting questions from 𝒬 i \mathcal{Q}_{i} , updating proficiency estimates after each step, and then accessing estimate’s precision by predicting responses on ℳ i \mathcal{M}_{i} . The assumption is that better score predictions reflect more accurate proficiency estimates. Thus, the binary classification metrics can be used for evaluations, e.g., Prediction Accuracy (ACC) and Area Under ROC Curve (AUC) [ 175 ] .

 
 
 Datasets. To evaluate the effectiveness and generalizability of a CAT system, it is crucial to use diverse datasets that not only challenge the algorithm but also reflect real-world testing scenarios. Such datasets typically contain the question bank, examinee response data, and relevant contextual data. Each of these components is essential for validating the CAT system itself in realistic settings. Three types of data can be used for this evaluation:

 
 
 (1) Human Educational Data: 
This category includes data collected from educational environments in practice, such as schools, universities, and online learning platforms. It provides insights into how examinees interact with educational content and assessments in a natural setting. The data may encompass examinee information, performance responses, learning behaviors, question characteristics, etc. We
have open-sourced a comprehensive education-related dataset library: https://github.com/bigdata-ustc/EduData . It includes a range of publicly available datasets along with previously private datasets, e.g., ASSISTments [ 176 ] , Junyi [ 177 ] , EdNet [ 178 ] , and Eedi2020 [ 179 ] . Additionally, we have provided a detailed data analysis to support further CAT research and application in educational settings, which can be found at the EduData GitHub link..

 
 
 (2) AI Model Response Data: The CAT paradigm is playing a crucial role in the evaluation of AI models. In particular, proficiency estimates are used to assess performance and rank models, especially for contemporary LLM evaluations. Various large-scale benchmarks and their corresponding response data can be utilized to build and test CAT systems, such as Google’s BIG-bench [ 180 ] , HuggingFace’s Open LLM Leaderboard [ 181 ] , HELM [ 182 ] , and AlpacaEval [ 183 ] . These benchmarks encompass a wide range of tasks, with topics spanning linguistics, mathematics, medicine, common-sense reasoning, biology, physics, social bias, programming, and beyond.

 
 
 (3) Simulated Datasets. These are artificially created datasets that mimic the characteristics of real examinee responses as illustrated above. They can be tailored to include specific patterns, noise levels, and distributions, allowing for controlled testing of the CAT system under various scenarios. Monte Carlo simulations can also be used to generate datasets with known properties and ground truth [ 174 ] . These datasets are useful for validating the CAT system’s capability to estimate proficiencies accurately and to adapt to the simulated changes during the testing process.

 
 
 

## IX Opportunities for Future Research 

 
 The integration of machine learning into CAT is poised to revolutionize the field. This section explores the future potential of machine learning to expand the applicability, interpretability, and multi-dimensionality of CAT systems.

 
 
 Multi-Dimensionality of the Assessment Process. Future research should harness machine learning to enhance the multi-dimensionality of the assessment. This involves not only the traditional response patterns but also the nuanced analysis of process data such as response times and mouse movements, which can provide insights into an examinee’s problem-solving strategies and levels of engagement. Moreover, integrating learning data, such as the examinee’s prior interactions with materials, can offer a longitudinal perspective on their learning trajectory and readiness for new concepts. Additionally, the analysis of content, encompassing textual, visual, and auditory materials [ 184 , 185 ] , allows for a richer understanding of how examinees interact with multifaceted information. Such machine learning-driven approaches promise to refine CAT systems comprehensively, enabling them to deliver assessments that are not just accurate reflections of an examinee’s proficiency but also predictive of their potential for future learning.

 
 
 Towards Explainable Machine Learning in CAT. Traditional CAT systems, particularly those based on information and statistic approaches, are lauded for their interpretability, from the parameters of measurement models to the logic behind the question selection algorithm. This transparency provides valuable insights to all stakeholders, including examinees, parents, and educators, and supports developers in debugging and refining the CAT system. In practice, interpretability is often important for deploying CAT in high-stakes settings: test providers may need to explain why certain items were chosen, demonstrate fair treatment across groups, and support audits or appeals. Even if the selection policy is complex, this can be addressed by making the main constraints transparent and recording a simple, human-readable reason for each selection. 
However, recent machine learning approaches, especially those employing deep learning, have an overwhelming advantage in capabilities on knowledge discovery, at the cost of reduced interpretability. Bridging the gap between these paradigms to create CAT systems that are both accurate and self-explanatory is a significant challenge that future research must address. This is particularly crucial for high-stakes standardized testing, where the outcomes carry significant consequences.

 
 
 Empowering CAT with Generative AI. Generative artificial intelligence (e.g., LLMs) is trained on massive, cross-domain datasets, endowing it with versatility and a profound repository of world knowledge [ 186 , 187 ] . These models have already shown preliminary progress in user modeling, such as recommendation systems, and in the generation of personalized strategy [ 188 , 189 ] . This connection is conceptually aligned with CAT: both aim to infer latent user traits (e.g., proficiency) from observed behavior and then adapt subsequent interactions accordingly. LLMs/agents can enrich the observation space beyond binary correctness by leveraging intermediate steps, explanations, hesitation patterns, and error types, which can support finer-grained proficiency estimation when properly calibrated. 
In the future, there is potential for these large models to significantly enhance CAT systems in various aspects, such as question selection, proficiency assessment, and even the automatic generation of novel, tailored questions on the fly [ 190 , 191 ] – questions that are not pre-existing in the bank. A practical integration is to use LLMs as assistive modules: they can draft candidate items conditioned on a targeted construct/skill label, format constraints, and an intended difficulty region, and produce useful metadata (topic tags, expected solution outlines, and common misconceptions) that helps index and retrieve items efficiently. We can envision a future where testing paradigms evolve towards greater intelligence and automation. A well-trained testing agent could engage with examinees in natural language interactions, utilizing various cues and process details to conduct a comprehensive assessment of abilities. This approach would move beyond the monotonous task of having examinees respond to questions from a predefined bank or benchmark one by one. Such advancements could lead to more effective and personalized testing experiences.

 
 
 Improving Machine Intelligence Evaluation. Traditional AI model evaluation relies extensively on large, gold-standard benchmarks. The maxim “more is better” has driven the use of larger benchmarks to provide comprehensive assessments. However, the sheer size of these benchmarks incurs significant time and computational costs, making fast and economical evaluations challenging. For example, evaluating the performance of a single LLM on the full HELM benchmark can consume over 4,000 GPU hours (or cost over $10,000 for APIs) [ 182 ] . Moreover, these benchmarks are often plagued by low-quality questions, errors, and contamination issues [ 40 ] . As discussed, an increasing number of researchers are attempting to leverage CAT and psychometrics to identify and address these issues, reducing evaluation overhead and gradually transforming it into a new evaluation paradigm. This shift is especially valuable as AI systems approach human-level performance, where CAT can offer finer-grained analysis of cognitive-like behaviors. 
 Although advanced LLMs differ fundamentally from humans in architecture, their learned behaviors often exhibit similar characteristics, since they are trained on large-scale human-produced data and display cognitive-like signatures [ 11 ] . CAT does not assume models are “human”; it only requires observable responses that can be consistently scored and related to item statistics. Ultimately, this emerging paradigm may lead to smarter, faster, and more cost-effective evaluations: deepening our understanding of both human and machine intelligence.

 
 
 

## X Conclusion 

 
 Computerized Adaptive Testing (CAT) has evolved over more than five decades, achieving remarkable progress in the intelligent evaluation of both humans and AI models through the support of statistical learning. In the past five years, the growing integration of deep learning into CAT has led to the emergence of innovative approaches that were previously unimaginable. These include algorithms for question selection learned directly from large-scale data, retrieval-based methods that improve selection efficiency by up to 200 × \times , and theoretical investigations into the upper bounds of estimation error. Although many of these methods are still in the early stages and not widely used in practice yet, they clearly point to a promising future for smarter testing systems powered by today’s wave of AI. 

 
 
 This comprehensive survey has highlighted the intricate and expansive nature of CAT, emphasizing the potential and prospects of integrating machine learning to enhance CAT systems. The paper primarily focused on the dual concerns of accuracy and efficiency within machine/human assessment. The insights presented are accessible and relevant not only to specialists in education and psychometrics but also to a broad spectrum of researchers. We encourage interested readers to explore the transformative impact of machine learning in this field and to use this survey as a reference for future research.

 
 
 

## References

 
 [1] 
 N. Mehrabi, F. Morstatter, N. Saxena, K. Lerman, and A. Galstyan (2021) 
 
 A survey on bias and fairness in machine learning .
 
 ACM computing surveys (CSUR) 54 ( 6 ), pp. 1–35 .
 
 Cited by: §I .
 

 [2] 
 Y. Rong, T. Leemann, T. Nguyen, L. Fiedler, P. Qian, V. Unhelkar, T. Seidel, G. Kasneci, and E. Kasneci (2023) 
 
 Towards human-centered explainable ai: a survey of user studies for model explanations .
 
 IEEE transactions on pattern analysis and machine intelligence .
 
 Cited by: §I .
 

 [3] 
 S. J. Chen, A. Choi, and A. Darwiche (2015) 
 
 Computer adaptive testing using the same-decision probability. .
 
 In BMA@ UAI ,
 
 pp. 34–43 .
 
 Cited by: §I .
 

 [4] 
 J. Vie, F. Popineau, É. Bruillard, and Y. Bourda (2017) 
 
 A review of recent advances in adaptive assessment .
 
 Learning analytics: fundaments, applications, and trends , pp. 113–142 .
 
 Cited by: §I ,
 §I ,
 §VIII .
 

 [5] 
 D. R. Eignor, M. L. Stocking, W. D. Way, and M. Steffen (1993) 
 
 CASE studies in computer adaptive test design through simulation 1, 2 .
 
 ETS Research Report Series 1993 ( 2 ), pp. i–41 .
 
 Cited by: §I .
 

 [6] 
 R. M. Luecht and R. J. Nungester (1998) 
 
 Some practical examples of computer-adaptive sequential testing .
 
 Journal of Educational Measurement 35 ( 3 ), pp. 229–249 .
 
 Cited by: §I ,
 §II .
 

 [7] 
 N. Otani, T. Nakazawa, D. Kawahara, and S. Kurohashi (2016) 
 
 IRT-based aggregation model of crowdsourced pairwise comparison for evaluating machine translations .
 
 In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing ,
 
 pp. 511–520 .
 
 Cited by: §I .
 

 [8] 
 J. P. Lalor, H. Wu, and H. Yu (2016) 
 
 Building an evaluation scale using item response theory .
 
 In Proceedings of the Conference on Empirical Methods in Natural Language Processing. Conference on Empirical Methods in Natural Language Processing ,
 
 Vol. 2016 , pp. 648 .
 
 Cited by: §I ,
 §II .
 

 [9] 
 J. Sedoc and L. Ungar (2020) 
 
 Item response theory for efficient human evaluation of chatbots .
 
 In Proceedings of the First Workshop on Evaluation and Comparison of NLP Systems ,
 
 pp. 21–33 .
 
 Cited by: §I .
 

 [10] 
 X. Wang, L. Jiang, J. Hernandez-Orallo, D. Stillwell, L. Sun, F. Luo, and X. Xie (2023) 
 
 Evaluating general-purpose ai with psychometrics .
 
 External Links: 2310.16379 
 
 Cited by: §I .
 

 [11] 
 Y. Zhuang, Q. Liu, Z. Pardos, P. C. Kyllonen, J. Zu, Z. Huang, S. Wang, and E. Chen (2025) 
 
 Position: AI evaluation should learn from how we test humans .
 
 In Forty-second International Conference on Machine Learning Position Paper Track ,
 
 Cited by: §I ,
 §II ,
 §IX .
 

 [12] 
 A. Kipnis, K. Voudouris, L. M. S. Buschoff, and E. Schulz (2024) 
 
 Metabench–a sparse benchmark to measure general ability in large language models .
 
 arXiv preprint arXiv:2407.12844 .
 
 Cited by: §I ,
 §V-A .
 

 [13] 
 T. A. Ackerman, M. J. Gierl, and C. M. Walker (2003) 
 
 Using multidimensional item response theory to evaluate educational and psychological tests .
 
 Educational Measurement: Issues and Practice 22 ( 3 ), pp. 37–51 .
 
 Cited by: §I ,
 §III-A .
 

 [14] 
 F. M. Lord (2012) 
 
 Applications of item response theory to practical testing problems .
 
 Routledge .
 
 Cited by: TABLE II ,
 TABLE III ,
 §I ,
 §III-A ,
 §V-A ,
 3rd item .
 

 [15] 
 H. Chang and Z. Ying (1996) 
 
 A global information approach to computerized adaptive testing .
 
 Applied Psychological Measurement 20 ( 3 ), pp. 213–229 .
 
 Cited by: TABLE III ,
 §I ,
 §III-A ,
 §V-A .
 

 [16] 
 H. Bi, H. Ma, Z. Huang, Y. Yin, Q. Liu, E. Chen, Y. Su, and S. Wang (2020) 
 
 Quality meets diversity: a model-agnostic framework for computerized adaptive testing .
 
 In 2020 IEEE International Conference on Data Mining (ICDM) ,
 
 pp. 42–51 .
 
 Cited by: TABLE II ,
 TABLE III ,
 §I ,
 §V-B ,
 §VII-A ,
 §VIII .
 

 [17] 
 H. Chang (2015) 
 
 Psychometrics behind computerized adaptive testing .
 
 Psychometrika 80 ( 1 ), pp. 1–20 .
 
 Cited by: §I ,
 §III ,
 §V-A ,
 §V-A .
 

 [18] 
 Y. Cheng (2008) 
 
 Computerized adaptive testing—new developments and applications .
 
 University of Illinois at Urbana-Champaign .
 
 Cited by: §I ,
 §V-A .
 

 [19] 
 D. F. Mujtaba and N. R. Mahapatra (2020) 
 
 Artificial intelligence in computerized adaptive testing .
 
 In 2020 International Conference on Computational Science and Computational Intelligence (CSCI) ,
 
 pp. 649–654 .
 
 Cited by: §I .
 

 [20] 
 B. Mirzasoleiman, J. Bilmes, and J. Leskovec (2020) 
 
 Coresets for data-efficient training of machine learning models .
 
 In Proceedings of the 37th International Conference on Machine Learning , H. D. III and A. Singh (Eds.) ,
 
 Proceedings of Machine Learning Research , Vol. 119 , pp. 6950–6960 .
 
 Cited by: §I .
 

 [21] 
 Y. Zhuang, Q. Liu, G. Zhao, Z. Huang, W. Huang, Z. Pardos, E. Chen, J. Wu, and X. Li (2023) 
 
 A bounded ability estimation for computerized adaptive testing .
 
 In Thirty-seventh Conference on Neural Information Processing Systems ,
 
 Cited by: TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 §I ,
 §III-A ,
 Fig. 6 ,
 §V-E .
 

 [22] 
 F. Wang, Q. Liu, E. Chen, Z. Huang, Y. Yin, S. Wang, and Y. Su (2023) 
 
 NeuralCD: a general framework for cognitive diagnosis .
 
 IEEE Transactions on Knowledge and Data Engineering 35 ( 8 ), pp. 8312–8327 .
 
 External Links: Document 
 
 Cited by: TABLE III ,
 TABLE III ,
 §I ,
 §IV-C .
 

 [23] 
 Z. Qiu, X. Wu, and W. Fan (2019) 
 
 Question difficulty prediction for multiple choice problems in medical exams .
 
 In Proceedings of the 28th ACM International Conference on Information and Knowledge Management ,
 
 pp. 139–148 .
 
 Cited by: §I ,
 §VI-A .
 

 [24] 
 Y. Zhuang, Q. Liu, Z. Huang, Z. Li, S. Shen, and H. Ma (2022) 
 
 Fully adaptive framework: neural computerized adaptive testing for online education .
 
 Proceedings of the AAAI Conference on Artificial Intelligence 36 ( 4 ), pp. 4734–4742 .
 
 Cited by: TABLE III ,
 §I ,
 §II ,
 1st item ,
 4th item ,
 §V-C ,
 §V-D .
 

 [25] 
 X. Li (2020) 
 
 Data-driven adaptive learning systems .
 
 Ph.D. Thesis .
 
 Cited by: §I ,
 §V-B .
 

 [26] 
 A. Ghosh and A. Lan (2021) 
 
 BOBCAT: bilevel optimization-based computerized adaptive testing .
 
 In Proceedings of the Thirtieth International Joint Conference on
Artificial Intelligence, IJCAI-21 ,
 
 pp. 2410–2417 .
 
 Cited by: TABLE III ,
 §I ,
 Fig. 5 ,
 §V-B ,
 §V-D .
 

 [27] 
 H. Wainer, N. J. Dorans, R. Flaugher, B. F. Green, and R. J. Mislevy (2000) 
 
 Computerized adaptive testing: a primer .
 
 Routledge .
 
 Cited by: §II .
 

 [28] 
 W. A. Sands, B. K. Waters, and J. R. McBride (1997) 
 
 Computerized adaptive testing: from inquiry to operation. .
 
 American Psychological Association .
 
 Cited by: §II .
 

 [29] 
 E. E. Roskam and P. G. Jansen (1984) 
 
 A new derivation of the rasch model .
 
 In Advances in Psychology ,
 
 Vol. 20 , pp. 293–307 .
 
 Cited by: §II .
 

 [30] 
 A. J. Verschoor and G. J. Straetmans (2010) 
 
 MATHCAT: a flexible testing system in mathematics education for adults .
 
 Elements of adaptive testing , pp. 137–149 .
 
 Cited by: §II .
 

 [31] 
 H. Wainer and G. L. Kiely (1987) 
 
 Item clusters and computerized adaptive testing: a case for testlets .
 
 Journal of Educational measurement 24 ( 3 ), pp. 185–201 .
 
 Cited by: §II .
 

 [32] 
 R. D. Gibbons, D. J. Weiss, E. Frank, and D. Kupfer (2016) 
 
 Computerized adaptive diagnosis and testing of mental health disorders .
 
 Annual review of clinical psychology 12 ( 1 ), pp. 83–104 .
 
 Cited by: §II .
 

 [33] 
 R. D. Gibbons, G. Hooker, M. D. Finkelman, D. J. Weiss, P. A. Pilkonis, E. Frank, T. Moore, and D. J. Kupfer (2013) 
 
 The computerized adaptive diagnostic test for major depressive disorder (cad-mdd): a screening tool for depression .
 
 The Journal of clinical psychiatry 74 ( 7 ), pp. 3579 .
 
 Cited by: §II .
 

 [34] 
 R. D. Gibbons, D. Kupfer, E. Frank, T. Moore, D. G. Beiser, and E. D. Boudreaux (2017) 
 
 Development of a computerized adaptive test suicide scale—the cat-ss .
 
 The Journal of clinical psychiatry 78 ( 9 ), pp. 3581 .
 
 Cited by: §II .
 

 [35] 
 J. M. Montgomery and J. Cutler (2013) 
 
 Computerized adaptive testing for public opinion surveys .
 
 Political Analysis 21 ( 2 ), pp. 172–192 .
 
 Cited by: §II .
 

 [36] 
 K. Ando, S. Mishio, and T. Nishijima (2018) 
 
 Validity and reliability of computerized adaptive test of soccer tactical skill .
 
 Football Science 15 , pp. 38–51 .
 
 Cited by: §II .
 

 [37] 
 M. Yurtcu and C. GÜZELLER (2021) 
 
 Bibliometric analysis of articles on computerized adaptive testing .
 
 Participatory Educational Research 8 ( 4 ), pp. 426–438 .
 
 Cited by: §II .
 

 [38] 
 F. M. Polo, L. Weber, L. Choshen, Y. Sun, G. Xu, and M. Yurochkin (2024) 
 
 TinyBenchmarks: evaluating llms with fewer examples .
 
 In Forty-first International Conference on Machine Learning ,
 
 Cited by: §II ,
 §IV-A ,
 §V-E .
 

 [39] 
 G. Guinet, B. Omidvar-Tehrani, A. Deoras, and L. Callot 
 
 Automated evaluation of retrieval-augmented language models with task-specific exam generation .
 
 In Forty-first International Conference on Machine Learning ,
 
 Cited by: §II .
 

 [40] 
 P. Rodriguez, J. Barrow, A. M. Hoyle, J. P. Lalor, R. Jia, and J. Boyd-Graber (2021) 
 
 Evaluation examples are not equally informative: how should that change nlp leaderboards? .
 
 In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) ,
 
 pp. 4486–4503 .
 
 Cited by: §II ,
 §IX .
 

 [41] 
 F. Martínez-Plumed, R. B.C. Prudêncio, A. Martínez-Usó, and J. Hernández-Orallo (2019) 
 
 Item response theory in ai: analysing machine learning classifiers at the instance level .
 
 Artificial Intelligence 271 , pp. 18–42 .
 
 External Links: ISSN 0004-3702 
 
 Cited by: §II .
 

 [42] 
 F. Martínez-Plumed, R. B. Prudêncio, A. Martínez-Usó, and J. Hernández-Orallo (2016) 
 
 Making sense of item response theory in machine learning .
 
 In ECAI 2016 ,
 
 pp. 1140–1148 .
 
 Cited by: §II .
 

 [43] 
 Y. Zheng, S. Nydick, S. Huang, and S. Zhang (2024) 
 
 MxML (exploring the relationship between measurement and machine learning): current state of the field .
 
 Educational Measurement: Issues and Practice 43 ( 1 ), pp. 19–38 .
 
 Cited by: §II .
 

 [44] 
 J. De La Torre (2009) 
 
 DINA model and parameter estimation: a didactic .
 
 Journal of educational and behavioral statistics 34 ( 1 ), pp. 115–130 .
 
 Cited by: §III-A ,
 §IV-B .
 

 [45] 
 Y. Cheng (2009) 
 
 When cognitive diagnosis meets computerized adaptive testing: cd-cat .
 
 Psychometrika 74 , pp. 619–632 .
 
 Cited by: §IV ,
 §V-A ,
 §VIII .
 

 [46] 
 S. E. Embretson and S. P. Reise (2013) 
 
 Item response theory .
 
 Psychology Press .
 
 Cited by: TABLE III ,
 TABLE III ,
 §IV-A .
 

 [47] 
 X. An and Y. Yung (2014) 
 
 Item response theory: what it is and how you can use the irt procedure to apply it .
 
 SAS Institute Inc. SAS364-2014 10 ( 4 ), pp. 1–14 .
 
 Cited by: §IV-A .
 

 [48] 
 D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt (2021) 
 
 Measuring massive multitask language understanding .
 
 In International Conference on Learning Representations ,
 
 Cited by: §IV-A .
 

 [49] 
 M. D. Reckase (2006) 
 
 18 multidimensional item response theory .
 
 Handbook of statistics 26 , pp. 607–642 .
 
 Cited by: §IV-A ,
 §V-A .
 

 [50] 
 M. Von Davier (2014) 
 
 The dina model as a constrained general diagnostic model: two variants of a model equivalency .
 
 British Journal of Mathematical and Statistical Psychology 67 ( 1 ), pp. 49–71 .
 
 Cited by: §IV-B .
 

 [51] 
 J. de la Torre (2011) 
 
 The generalized DINA model framework .
 
 Psychometrika 76 ( 2 ), pp. 179–199 .
 
 Note: Place: Germany
Publisher: Springer 
 
 External Links: ISSN 1860-0980 ,
 Document 
 
 Cited by: §IV-B .
 

 [52] 
 Q. Liu, R. Wu, E. Chen, G. Xu, Y. Su, Z. Chen, and G. Hu (2018) 
 
 Fuzzy cognitive diagnosis for modelling examinee performance .
 
 ACM Transactions on Intelligent Systems and Technology (TIST) 9 ( 4 ), pp. 1–26 .
 
 Cited by: §IV-B .
 

 [53] 
 J. P. Leighton, M. J. Gierl, and S. M. Hunka (2004) 
 
 The attribute hierarchy method for cognitive assessment: a variation on tatsuoka’s rule-space approach .
 
 Journal of educational measurement 41 ( 3 ), pp. 205–237 .
 
 Cited by: §IV-B .
 

 [54] 
 S. Cheng, Q. Liu, E. Chen, Z. Huang, Z. Huang, Y. Chen, H. Ma, and G. Hu (2019) 
 
 DIRT: deep learning enhanced item response theory for cognitive diagnosis .
 
 In Proceedings of the 28th ACM International Conference on Information and Knowledge Management ,
 
 pp. 2397–2400 .
 
 Cited by: §IV-C .
 

 [55] 
 W. Gao, Q. Liu, Z. Huang, Y. Yin, H. Bi, M. Wang, J. Ma, S. Wang, and Y. Su (2021) 
 
 RCD: relation map driven cognitive diagnosis for intelligent education systems .
 
 In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval ,
 
 pp. 501–510 .
 
 Cited by: §IV-C .
 

 [56] 
 J. Li, F. Wang, Q. Liu, M. Zhu, W. Huang, Z. Huang, E. Chen, Y. Su, and S. Wang (2022) 
 
 HierCDF: a bayesian network-based hierarchical cognitive diagnosis framework .
 
 In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining ,
 
 pp. 904–913 .
 
 Cited by: §IV-C .
 

 [57] 
 L. Gao, Z. Zhao, C. Li, J. Zhao, and Q. Zeng (2022) 
 
 Deep cognitive diagnosis model for predicting students’ performance .
 
 Future Generation Computer Systems 126 , pp. 252–262 .
 
 External Links: ISSN 0167-739X 
 
 Cited by: §IV-C .
 

 [58] 
 J. J. Rissanen (1996) 
 
 Fisher information and stochastic complexity .
 
 IEEE transactions on information theory 42 ( 1 ), pp. 40–47 .
 
 Cited by: §V-A .
 

 [59] 
 S. M. Ross (2014) 
 
 A first course in probability .
 
 Pearson .
 
 Cited by: Theorem 1 .
 

 [60] 
 G. Hooker, M. Finkelman, and A. Schwartzman (2009) 
 
 Paradoxical results in multidimensional item response theory .
 
 Psychometrika 74 ( 3 ), pp. 419–442 .
 
 Cited by: §V-A .
 

 [61] 
 W. J. Veerkamp and M. P. Berger (1997) 
 
 Some new item selection criteria for adaptive testing .
 
 Journal of Educational and Behavioral Statistics 22 ( 2 ), pp. 203–226 .
 
 Cited by: §V-A .
 

 [62] 
 W. J. van der Linden (1998) 
 
 Bayesian item selection criteria for adaptive testing .
 
 Psychometrika 63 ( 2 ), pp. 201–216 .
 
 Cited by: §V-A .
 

 [63] 
 J. R. Barrada, F. J. Abad, and B. P. Veldkamp (2009) 
 
 METODOLOGÍa: comparison of methods for controlling maximum exposure rates in computerized adaptive testing .
 
 Psicothema , pp. 313–320 .
 
 Cited by: §V-A .
 

 [64] 
 J. R. Barrada, P. Mazuela, and J. Olea (2006) 
 
 Maximum information stratification method for controlling item exposure in computerized adaptive testing .
 
 Psicothema 18 ( 1 ), pp. 156–159 .
 
 Cited by: §V-A ,
 §VII-A .
 

 [65] 
 Y. Zheng, H. Cheon, and C. M. Katz (2020) 
 
 Using machine learning methods to develop a short tree-based adaptive classification test: case study with a high-dimensional item pool and imbalanced data .
 
 Applied psychological measurement 44 ( 7-8 ), pp. 499–514 .
 
 Cited by: §V-A .
 

 [66] 
 R. Henson and J. Douglas (2005) 
 
 Test construction for cognitive diagnosis .
 
 Applied Psychological Measurement 29 ( 4 ), pp. 262–277 .
 
 Cited by: §V-A .
 

 [67] 
 C. Tatsuoka (2002) 
 
 Data analytic methods for latent partially ordered classification models .
 
 Journal of the Royal Statistical Society Series C: Applied Statistics 51 ( 3 ), pp. 337–350 .
 
 Cited by: §V-A .
 

 [68] 
 H. Kang, S. Zhang, and H. Chang (2017) 
 
 Dual-objective item selection criteria in cognitive diagnostic computerized adaptive testing .
 
 Journal of Educational Measurement 54 ( 2 ), pp. 165–183 .
 
 Cited by: §V-A .
 

 [69] 
 C. Zheng, G. He, and C. Gao (2018) 
 
 The information product methods: a unified approach to dual-purpose computerized adaptive testing .
 
 Applied Psychological Measurement 42 ( 4 ), pp. 321–324 .
 
 Cited by: §V-A .
 

 [70] 
 B. Dai, M. Zhang, and G. Li (2016) 
 
 Exploration of item selection in dual-purpose cognitive diagnostic computerized adaptive testing: based on the rrum .
 
 Applied Psychological Measurement 40 ( 8 ), pp. 625–640 .
 
 Cited by: §V-A .
 

 [71] 
 A. Krishnakumar (2007) 
 
 Active learning literature survey .
 
 pp.  .
 
 Cited by: §V-B .
 

 [72] 
 S. Huang, R. Jin, and Z. Zhou (2014) 
 
 Active learning by querying informative and representative examples .
 
 IEEE Transactions on Pattern Analysis and Machine Intelligence 36 ( 10 ), pp. 1936–1949 .
 
 External Links: Document 
 
 Cited by: §V-B .
 

 [73] 
 D. Yoo and I. S. Kweon (2019) 
 
 Learning loss for active learning .
 
 In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition ,
 
 pp. 93–102 .
 
 Cited by: §V-B .
 

 [74] 
 A. Ghorbani, J. Zou, and A. Esteva (2022) 
 
 Data shapley valuation for efficient batch active learning .
 
 In 2022 56th Asilomar Conference on Signals, Systems, and Computers ,
 
 pp. 1456–1462 .
 
 Cited by: §V-B .
 

 [75] 
 J. Li, P. Chen, S. Yu, S. Liu, and J. Jia (2024) 
 
 BAL: balancing diversity and novelty for active learning .
 
 IEEE Transactions on Pattern Analysis and Machine Intelligence 46 ( 5 ), pp. 3653–3664 .
 
 External Links: Document 
 
 Cited by: §V-B .
 

 [76] 
 S. Wang, Y. Li, K. Ma, R. Ma, H. Guan, and Y. Zheng (2020) 
 
 Dual adversarial network for deep active learning .
 
 In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXIV 16 ,
 
 pp. 680–696 .
 
 Cited by: §V-B .
 

 [77] 
 R. S. Sutton and A. G. Barto (2018) 
 
 Reinforcement learning: an introduction .
 
 MIT press .
 
 Cited by: §V-C ,
 §V-C .
 

 [78] 
 K. Arulkumaran, M. P. Deisenroth, M. Brundage, and A. A. Bharath (2017) 
 
 A brief survey of deep reinforcement learning .
 
 arXiv preprint arXiv:1708.05866 .
 
 Cited by: §V-C .
 

 [79] 
 J. Kober, J. A. Bagnell, and J. Peters (2013) 
 
 Reinforcement learning in robotics: a survey .
 
 The International Journal of Robotics Research 32 ( 11 ), pp. 1238–1274 .
 
 Cited by: §V-C .
 

 [80] 
 E. A. Feinberg and A. Shwartz (2012) 
 
 Handbook of markov decision processes: methods and applications .
 
 Vol. 40 , Springer Science Business Media .
 
 Cited by: §V-C .
 

 [81] 
 A. Agarwal, S. M. Kakade, J. D. Lee, and G. Mahajan (2021) 
 
 On the theory of policy gradient methods: optimality, approximation, and distribution shift .
 
 The Journal of Machine Learning Research 22 ( 1 ), pp. 4431–4506 .
 
 Cited by: §V-C .
 

 [82] 
 V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu (2016) 
 
 Asynchronous methods for deep reinforcement learning .
 
 In International conference on machine learning ,
 
 pp. 1928–1937 .
 
 Cited by: §V-C .
 

 [83] 
 X. Li, H. Xu, J. Zhang, and H. Chang (2020) 
 
 Deep reinforcement learning for adaptive learning systems .
 
 arXiv preprint arXiv:2004.08410 .
 
 Cited by: 1st item .
 

 [84] 
 D. Nurakhmetov (2019) 
 
 Reinforcement learning applied to adaptive classification testing .
 
 Theoretical and Practical Advances in Computer-based Educational Measurement , pp. 325–336 .
 
 Cited by: 1st item ,
 4th item ,
 §V-C .
 

 [85] 
 H. Ma, Y. Zeng, S. Yang, C. Qin, X. Zhang, and L. Zhang (2023) 
 
 A novel computerized adaptive testing framework with decoupled learning selector .
 
 Complex Intelligent Systems , pp. 1–12 .
 
 Cited by: TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 4th item ,
 §V-D .
 

 [86] 
 H. Wang, T. Long, L. Yin, W. Zhang, W. Xia, Q. Hong, D. Xia, R. Tang, and Y. Yu (2023) 
 
 GMOCAT: a graph-enhanced multi-objective method for computerized adaptive testing .
 
 In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining ,
 
 pp. 2279–2289 .
 
 Cited by: TABLE II ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 4th item ,
 §V-D ,
 §VII-A .
 

 [87] 
 J. Shin and O. Bulut (2022) 
 
 Building an intelligent recommendation system for personalized test scheduling in computerized assessments: a reinforcement learning approach .
 
 Behavior Research Methods 54 ( 1 ), pp. 216–232 .
 
 Cited by: 4th item .
 

 [88] 
 X. Li, H. Xu, J. Zhang, and H. Chang (2023) 
 
 Deep reinforcement learning for adaptive learning systems .
 
 Journal of Educational and Behavioral Statistics 48 ( 2 ), pp. 220–243 .
 
 Cited by: §V-C .
 

 [89] 
 D. P. Bertsekas and J. N. Tsitsiklis (1991) 
 
 An analysis of stochastic shortest path problems .
 
 Mathematics of Operations Research 16 ( 3 ), pp. 580–595 .
 
 Cited by: §V-C .
 

 [90] 
 P. Gilavert and V. Freire (2022) 
 
 Computerized adaptive testing: a unified approach under markov decision process .
 
 In International Conference on Computational Science and Its Applications ,
 
 pp. 591–602 .
 
 Cited by: §V-C ,
 §V-C .
 

 [91] 
 F. Doshi-Velez, D. Pfau, F. Wood, and N. Roy (2015) 
 
 Bayesian nonparametric methods for partially-observable reinforcement learning .
 
 IEEE Transactions on Pattern Analysis and Machine Intelligence 37 ( 2 ), pp. 394–407 .
 
 External Links: Document 
 
 Cited by: §V-C .
 

 [92] 
 Y. Zhuang, Q. Liu, Z. Huang, Z. Li, B. Jin, H. Bi, E. Chen, and S. Wang (2022) 
 
 A robust computerized adaptive testing approach in educational question retrieval .
 
 In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval ,
 
 pp. 416–426 .
 
 Cited by: TABLE II ,
 TABLE II ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 §V-C ,
 §VII-C .
 

 [93] 
 E. Drousiotis, P. Pentaliotis, L. Shi, and A. I. Cristea (2021) 
 
 Capturing fairness and uncertainty in student dropout prediction–a comparison study .
 
 In International Conference on Artificial Intelligence in Education ,
 
 pp. 139–144 .
 
 Cited by: §V-C .
 

 [94] 
 Y. Chen, X. Li, J. Liu, and Z. Ying (2018) 
 
 Recommendation system for adaptive learning .
 
 Applied psychological measurement 42 ( 1 ), pp. 24–41 .
 
 Cited by: §V-C .
 

 [95] 
 M. Hoerger and H. Kurniawati (2021) 
 
 An on-line pomdp solver for continuous observation spaces .
 
 In 2021 IEEE International Conference on Robotics and Automation (ICRA) ,
 
 pp. 7643–7649 .
 
 Cited by: §V-C .
 

 [96] 
 J. Schwartz, R. Zhou, and H. Kurniawati (2022) 
 
 Online planning for interactive-pomdps using nested monte carlo tree search .
 
 In 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) ,
 
 pp. 8770–8777 .
 
 Cited by: §V-C .
 

 [97] 
 C. Finn, P. Abbeel, and S. Levine (2017) 
 
 Model-agnostic meta-learning for fast adaptation of deep networks .
 
 In International Conference on Machine Learning ,
 
 pp. 1126–1135 .
 
 Cited by: §V-D .
 

 [98] 
 Z. Xu, X. Chen, and L. Cao (2022) 
 
 Fast task adaptation based on the combination of model-based and gradient-based meta learning .
 
 IEEE Transactions on Cybernetics 52 ( 6 ), pp. 5209–5218 .
 
 External Links: Document 
 
 Cited by: §V-D .
 

 [99] 
 W. Feng, A. Ghosh, S. Sireci, and A. S. Lan (2023) 
 
 Balancing test accuracy and security in computerized adaptive testing .
 
 arXiv preprint arXiv:2305.18312 .
 
 Cited by: §V-D .
 

 [100] 
 J. Yu, M. Zhenyu, J. Lei, L. Yin, W. Xia, Y. Yu, and T. Long (2023) 
 
 SACAT: student-adaptive computerized adaptive testing .
 
 In The Fifth International Conference on Distributed Artificial Intelligence ,
 
 pp. 1–7 .
 
 Cited by: TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 TABLE III ,
 §V-D .
 

 [101] 
 A. Miller (2002) 
 
 Subset selection in regression .
 
 CRC Press .
 
 Cited by: §V-E .
 

 [102] 
 D. F. Mujtaba and N. R. Mahapatra (2021) 
 
 Multi-objective optimization of item selection in computerized adaptive testing .
 
 In Proceedings of the Genetic and Evolutionary Computation Conference ,
 
 pp. 1018–1026 .
 
 Cited by: §V-E .
 

 [103] 
 K. Deb (2011) 
 
 Multi-objective optimisation using evolutionary algorithms: an introduction .
 
 In Multi-objective evolutionary optimisation for product design and manufacturing ,
 
 pp. 3–34 .
 
 Cited by: §V-E .
 

 [104] 
 J. Yu, Y. Zhuang, Z. Huang, Q. Liu, X. Li, R. LI, and E. Chen (2024) 
 
 A unified adaptive testing system enabled by hierarchical structure search .
 
 In Forty-first International Conference on Machine Learning ,
 
 Cited by: §V-E .
 

 [105] 
 R. Conejo, E. Guzmán, E. Millán, M. Trella, J. L. Pérez-De-La-Cruz, and A. Ríos (2004) 
 
 SIETTE: a web-based tool for adaptive testing .
 
 International Journal of Artificial Intelligence in Education 14 ( 1 ), pp. 29–61 .
 
 Cited by: §VI-A .
 

 [106] 
 J. López-Cuadrado, A. Armendariz, and T. Pérez (2006) 
 
 Adaptive evaluation in an e-learning system architecture .
 
 Current Developments in Technology-Assisted Education , pp. 1507–1511 .
 
 Cited by: §VI-A .
 

 [107] 
 J. López-Cuadrado, A. Armendariz, T. A. Pérez, and R. Arruabarrena (2008) 
 
 Helping tools for item bank calibration and development of computerized adaptive tests .
 
 In International Technology, Education, and Development Conference (INTED2008). Valencia, España: International Association of Technology, Education, and Development ,
 
 Cited by: §VI-A .
 

 [108] 
 A. Kozierkiewicz-Hetmańska and R. Poniatowski (2014) 
 
 An item bank calibration method for a computer adaptive test .
 
 In Asian Conference on Intelligent Information and Database Systems ,
 
 pp. 375–383 .
 
 Cited by: §VI-A .
 

 [109] 
 Y. Liu, S. Bhandari, and Z. A. Pardos (2024) 
 
 Leveraging llm-respondents for item evaluation: a psychometric analysis .
 
 arXiv preprint arXiv:2407.10899 .
 
 Cited by: §VI-A .
 

 [110] 
 A. F. De Champlain (2010) 
 
 A primer on classical test theory and item response theory for assessments in medical education .
 
 Medical education 44 ( 1 ), pp. 109–117 .
 
 Cited by: §VI-A .
 

 [111] 
 C. Magno (2009) 
 
 Demonstrating the difference between classical test theory and item response theory using derived test data .
 
 The international Journal of Educational and Psychological assessment 1 ( 1 ), pp. 1–11 .
 
 Cited by: §VI-A .
 

 [112] 
 R. F. DeVellis (2006) 
 
 Classical test theory .
 
 Medical care , pp. S50–S59 .
 
 Cited by: §VI-A .
 

 [113] 
 W. Chang and H. Yang (2009) 
 
 Applying irt to estimate learning ability and k-means clustering in web based learning. .
 
 J. Softw. 4 ( 2 ), pp. 167–174 .
 
 Cited by: §VI-A .
 

 [114] 
 J. Liu, G. Xu, and Z. Ying (2013) 
 
 Theory of the self-learning q-matrix .
 
 Bernoulli: official journal of the Bernoulli Society for Mathematical Statistics and Probability 19 ( 5A ), pp. 1790 .
 
 Cited by: §VI-A .
 

 [115] 
 Y. Sun, S. Ye, S. Inoue, and Y. Sun (2014) 
 
 Alternating recursive method for q-matrix learning .
 
 In Educational Data Mining 2014 ,
 
 Cited by: §VI-A .
 

 [116] 
 J. Xiong, Z. Luo, G. Luo, and X. Yu (2022) 
 
 Data-driven q-matrix learning based on boolean matrix factorization in cognitive diagnostic assessment .
 
 British Journal of Mathematical and Statistical Psychology 75 ( 3 ), pp. 638–667 .
 
 Cited by: §VI-A .
 

 [117] 
 Z. Huang, Q. Liu, E. Chen, H. Zhao, M. Gao, S. Wei, Y. Su, and G. Hu (2017) 
 
 Question difficulty prediction for reading problems in standard tests .
 
 In Proceedings of the AAAI Conference on Artificial Intelligence ,
 
 Vol. 31 .
 
 Cited by: §VI-A .
 

 [118] 
 Y. Huang, W. Huang, S. Tong, Z. Huang, Q. Liu, E. Chen, J. Ma, L. Wan, and S. Wang (2021) 
 
 Stan: adversarial network for cross-domain question difficulty prediction .
 
 In 2021 IEEE International Conference on Data Mining (ICDM) ,
 
 pp. 220–229 .
 
 Cited by: §VI-A .
 

 [119] 
 W. Huang, E. Chen, Q. Liu, Y. Chen, Z. Huang, Y. Liu, Z. Zhao, D. Zhang, and S. Wang (2019) 
 
 Hierarchical multi-label text classification: an attention-based recurrent network approach .
 
 In Proceedings of the 28th ACM international conference on information and knowledge management ,
 
 pp. 1051–1060 .
 
 Cited by: §VI-A .
 

 [120] 
 W. Huang, E. Chen, Q. Liu, H. Xiong, Z. Huang, S. Tong, and D. Zhang (2022) 
 
 HmcNet: a general approach for hierarchical multi-label classification .
 
 IEEE Transactions on Knowledge and Data Engineering .
 
 Cited by: §VI-A .
 

 [121] 
 S. Lei, W. Huang, S. Tong, Q. Liu, Z. Huang, E. Chen, and Y. Su (2021) 
 
 Consistency-aware multi-modal network for hierarchical multi-label classification in online education system .
 
 In 2021 IEEE International Conference on Big Knowledge (ICBK) ,
 
 pp. 1–8 .
 
 Cited by: §VI-A .
 

 [122] 
 Y. Yin, Q. Liu, Z. Huang, E. Chen, W. Tong, S. Wang, and Y. Su (2019) 
 
 Quesnet: a unified representation for heterogeneous test questions .
 
 In Proceedings of the 25th acm sigkdd international conference on knowledge discovery data mining ,
 
 pp. 1328–1336 .
 
 Cited by: §VI-A .
 

 [123] 
 Y. Ning, Z. Huang, X. Lin, E. Chen, S. Tong, Z. Gong, and S. Wang (2023) 
 
 Towards a holistic understanding of mathematical questions with contrastive pre-training .
 
 arXiv preprint arXiv:2301.07558 .
 
 Cited by: §VI-A .
 

 [124] 
 J. Revuelta and V. Ponsoda (1998) 
 
 A comparison of item exposure control methods in computerized adaptive testing .
 
 Journal of Educational Measurement 35 ( 4 ), pp. 311–327 .
 
 Cited by: §VI-B .
 

 [125] 
 M. D. Reckase (2010) 
 
 Designing item pools to optimize the functioning of a computerized adaptive test .
 
 Psychological Test and Assessment Modeling 52 ( 2 ), pp. 127 .
 
 Cited by: §VI-B ,
 §VI-B .
 

 [126] 
 D. O. Segall (2005) 
 
 Computerized adaptive testing .
 
 Encyclopedia of social measurement 1 , pp. 429–438 .
 
 Cited by: §VI-B .
 

 [127] 
 W. He and M. D. Reckase (2014) 
 
 Item pool design for an operational variable-length computerized adaptive test .
 
 Educational and Psychological Measurement 74 ( 3 ), pp. 473–494 .
 
 Cited by: §VI-B .
 

 [128] 
 W. J. Van Der Linden, B. P. Veldkamp, and L. M. Reese (2000) 
 
 An integer-programming approach to item pool design. law school admission council computerized testing report. lsac research report series. .
 
 Cited by: §VI-B .
 

 [129] 
 W. D. Way, M. Steffen, and G. S. Anderson (2005) 
 
 Developing, maintaining, and renewing the item inventory to support cbt .
 
 In Computer-Based Testing ,
 
 pp. 143–164 .
 
 Cited by: §VI-B .
 

 [130] 
 W. J. van der Linden, A. Ariel, and B. P. Veldkamp (2006) 
 
 Assembling a computerized adaptive testing item pool as a set of linear tests .
 
 Journal of Educational and Behavioral Statistics 31 ( 1 ), pp. 81–99 .
 
 Cited by: §VI-B .
 

 [131] 
 M. L. Stocking and L. Swanson (1998) 
 
 Optimal design of item banks for computerized adaptive tests .
 
 Applied Psychological Measurement 22 ( 3 ), pp. 271–279 .
 
 Cited by: §VI-B .
 

 [132] 
 A. Ariel, B. P. Veldkamp, and W. J. van der Linden (2004) 
 
 Constructing rotating item pools for constrained adaptive testing .
 
 Journal of Educational Measurement 41 ( 4 ), pp. 345–359 .
 
 Cited by: §VI-B .
 

 [133] 
 L. Swanson and M. L. Stocking (1993) 
 
 A model and heuristic for solving very large item selection problems .
 
 Applied Psychological Measurement 17 ( 2 ), pp. 151–166 .
 
 Cited by: §VI-B .
 

 [134] 
 H. Gulliksen (2013) 
 
 Theory of mental tests .
 
 Routledge .
 
 Cited by: §VI-B .
 

 [135] 
 J. Sympson and R. Hetter (1985) 
 
 Controlling item-exposure rates in computerized adaptive testing .
 
 In Proceedings of the 27th annual meeting of the Military Testing Association ,
 
 pp. 973–977 .
 
 Cited by: TABLE II ,
 §VII-A .
 

 [136] 
 H. Chang and Z. Ying (1999) 
 
 A-stratified multistage computerized adaptive testing .
 
 Applied Psychological Measurement 23 ( 3 ), pp. 211–222 .
 
 Cited by: TABLE II ,
 §VII-A .
 

 [137] 
 W. J. van der Linden and B. P. Veldkamp (2004) 
 
 Constraining item exposure in computerized adaptive testing with shadow tests .
 
 Journal of Educational and Behavioral Statistics 29 ( 3 ), pp. 273–291 .
 
 Cited by: §VII-A .
 

 [138] 
 W. J. van der Linden and B. P. Veldkamp (2007) 
 
 Conditional item-exposure control in adaptive testing using item-ineligibility probabilities .
 
 Journal of Educational and Behavioral Statistics 32 ( 4 ), pp. 398–418 .
 
 Cited by: §VII-A .
 

 [139] 
 J. R. Barrada, B. P. Veldkamp, and J. Olea (2009) 
 
 Multiple maximum exposure rates in computerized adaptive testing .
 
 Applied Psychological Measurement 33 ( 1 ), pp. 58–73 .
 
 Cited by: TABLE II ,
 §VII-A .
 

 [140] 
 H. Chang, J. Qian, and Z. Ying (2001) 
 
 A-stratified multistage computerized adaptive testing with b blocking .
 
 Applied Psychological Measurement 25 ( 4 ), pp. 333–341 .
 
 Cited by: §VII-A .
 

 [141] 
 J. R. Barrada, F. J. Abad, and J. Olea (2014) 
 
 Optimal number of strata for the stratified methods in computerized adaptive testing .
 
 The Spanish Journal of Psychology 17 , pp. E48 .
 
 Cited by: TABLE II ,
 §VII-A .
 

 [142] 
 İ. Ü. Öcal and N. Doğan (2024) 
 
 Effect of content balancing on measurement precision in computer adaptive testing applications .
 
 Journal of Measurement and Evaluation in Education and Psychology 15 ( 4 ), pp. 395–407 .
 
 Cited by: §VII-A .
 

 [143] 
 C. Li and J. Flanigan (2024) 
 
 Task contamination: language models may not be few-shot anymore .
 
 In Proceedings of the AAAI Conference on Artificial Intelligence ,
 
 Vol. 38 , pp. 18471–18480 .
 
 Cited by: §VII-A .
 

 [144] 
 Y. Oren, N. Meister, N. S. Chatterji, F. Ladhak, and T. Hashimoto (2023) 
 
 Proving test set contamination for black-box language models .
 
 In The Twelfth International Conference on Learning Representations ,
 
 Cited by: §VII-A .
 

 [145] 
 T. A. Cleary (1968) 
 
 Test bias: prediction of grades of negro and white students in integrated colleges .
 
 Journal of Educational Measurement 5 , pp. 115–124 .
 
 Cited by: §VII-B .
 

 [146] 
 J. Chai and X. Wang (2022) 
 
 Fairness with adaptive weights .
 
 In Proceedings of the 39th International Conference on Machine Learning ,
 
 Vol. 162 , pp. 2853–2866 .
 
 Cited by: §VII-B .
 

 [147] 
 P. Li and H. Liu (2022) 
 
 Achieving fairness at no utility cost via data reweighing .
 
 In Proceedings of the 39th International Conference on Machine Learning ,
 
 Vol. 162 , pp. 12917–12930 .
 
 Cited by: §VII-B .
 

 [148] 
 J. Liu, J. Hou, N. Zhang, Z. Liu, and W. He (2022) 
 
 Learning evidential cognitive diagnosis networks robust to response bias .
 
 In CAAI International Conference on Artificial Intelligence ,
 
 pp. 171–181 .
 
 Cited by: TABLE II ,
 1st item .
 

 [149] 
 G. Thompson 
 
 Is the naplan results delay about politics or precision? .
 
 Note: https://blog.aare.edu.au/is-the-naplan-results-delay-about-politics-or-precision/ Accessed: 2022-8-29 
 
 Cited by: TABLE II ,
 1st item .
 

 [150] 
 R. F. Kizilcec and H. Lee (2022) 
 
 Algorithmic fairness in education .
 
 In The ethics of artificial intelligence in education ,
 
 pp. 174–202 .
 
 Cited by: TABLE II ,
 1st item .
 

 [151] 
 G. Camilli and L. A. Shepard (1994) 
 
 Methods for identifying biased test items .
 
 Vol. 4 , Sage .
 
 Cited by: TABLE II ,
 2nd item .
 

 [152] 
 R. K. Hambleton, H. Swaminathan, and H. J. Rogers (1991) 
 
 Fundamentals of item response theory .
 
 Vol. 2 , Sage .
 
 Cited by: TABLE II ,
 2nd item .
 

 [153] 
 P. Roberts 
 
 Standardised tests are culturally biased against rural students .
 
 Note: https://theconversation.com/standardised-tests-are-culturally-biased-against-rural-students-86305Accessed: 2017-11-21 
 
 Cited by: TABLE II ,
 2nd item .
 

 [154] 
 M. Chu and H. Lai (2013) 
 
 Detecting biased items using catsib to increase fairness in computer adaptive tests .
 
 Alberta Journal of Educational Research 59 ( 4 ), pp. 630–643 .
 
 Cited by: TABLE II ,
 2nd item .
 

 [155] 
 G. J. Mellenbergh (1989) 
 
 Item bias and item response theory .
 
 International journal of educational research 13 ( 2 ), pp. 127–143 .
 
 Cited by: 2nd item .
 

 [156] 
 B. F. Green, R. D. Bock, L. G. Humphreys, R. L. Linn, and M. D. Reckase (1984) 
 
 Technical guidelines for assessing computerized adaptive tests .
 
 Journal of Educational measurement 21 ( 4 ), pp. 347–360 .
 
 Cited by: TABLE II ,
 §VII-B .
 

 [157] 
 S. L. Brigman and W. Bashaw (1976) 
 
 Multiple test equating using the rasch model. .
 
 Cited by: §VII-B .
 

 [158] 
 W. J. van der Linden (2000) 
 
 A test-theoretic approach to observed-score equating .
 
 Psychometrika 65 ( 4 ), pp. 437–456 .
 
 Cited by: TABLE II ,
 §VII-B .
 

 [159] 
 E. JAN 
 
 Subpopulation differences in equating computerized adaptive and paper-and-pencil versions of the asvab .
 
 Cited by: TABLE II ,
 §VII-B .
 

 [160] 
 Y. Sawaki (2001) 
 
 Comparability of conventional and computerized tests of reading in a second language .
 
 Cited by: TABLE II ,
 §VII-B .
 

 [161] 
 F. B. Baker and S. Kim (2004) 
 
 Item response theory: parameter estimation techniques .
 
 CRC press .
 
 Cited by: TABLE II ,
 §VII-C .
 

 [162] 
 D. Li and H. Zhang (2021) 
 
 Improved regularization and robustness for fine-tuning in neural networks .
 
 Advances in Neural Information Processing Systems 34 , pp. 27249–27262 .
 
 Cited by: §VII-C .
 

 [163] 
 Y. Wang, G. Huang, S. Song, X. Pan, Y. Xia, and C. Wu (2021) 
 
 Regularizing deep networks with semantic data augmentation .
 
 IEEE Transactions on Pattern Analysis and Machine Intelligence 44 ( 7 ), pp. 3733–3748 .
 
 Cited by: §VII-C .
 

 [164] 
 Y. Dong, Q. Fu, X. Yang, T. Pang, H. Su, Z. Xiao, and J. Zhu (2020) 
 
 Benchmarking adversarial robustness on image classification .
 
 In proceedings of the IEEE/CVF conference on computer vision and pattern recognition ,
 
 pp. 321–331 .
 
 Cited by: §VII-C .
 

 [165] 
 S. Kariyappa and M. K. Qureshi (2019) 
 
 Improving adversarial robustness of ensembles with diversity training .
 
 arXiv preprint arXiv:1901.09981 .
 
 Cited by: §VII-C .
 

 [166] 
 B. P. Veldkamp and A. J. Verschoor (2019) 
 
 Robust computerized adaptive testing .
 
 Theoretical and practical advances in computer-based educational measurement , pp. 291–305 .
 
 Cited by: TABLE II ,
 §VII-C .
 

 [167] 
 L. M. Rudner (2009) 
 
 Implementing the graduate management admission test computerized adaptive test .
 
 In Elements of adaptive testing ,
 
 pp. 151–165 .
 
 Cited by: §VII-D .
 

 [168] 
 Y. Huang, Y. Lin, and S. Cheng (2009) 
 
 An adaptive testing system for supporting versatile educational assessment .
 
 Computers Education 52 ( 1 ), pp. 53–67 .
 
 Cited by: TABLE II ,
 1st item .
 

 [169] 
 C. Lee, M. Wang, C. Wang, O. Teytaud, J. Liu, S. Lin, and P. Hung (2018) 
 
 PSO-based fuzzy markup language for student learning performance evaluation and educational application .
 
 IEEE Transactions on Fuzzy Systems 26 ( 5 ), pp. 2618–2633 .
 
 External Links: Document 
 
 Cited by: 1st item .
 

 [170] 
 H. Zhu, X. Li, P. Zhang, G. Li, J. He, H. Li, and K. Gai (2018) 
 
 Learning tree-based deep model for recommender systems .
 
 In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery Data Mining ,
 
 pp. 1079–1088 .
 
 Cited by: 2nd item .
 

 [171] 
 S. Bao, Q. Xu, Z. Yang, X. Cao, and Q. Huang (2023) 
 
 Rethinking collaborative metric learning: toward an efficient alternative without negative sampling .
 
 IEEE Transactions on Pattern Analysis and Machine Intelligence 45 ( 1 ), pp. 1017–1035 .
 
 External Links: Document 
 
 Cited by: 2nd item .
 

 [172] 
 Y. Hong, S. Tong, W. Huang, Y. Zhuang, Q. Liu, E. Chen, X. Li, and Y. He (2023) 
 
 Search-efficient computerized adaptive testing .
 
 In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management ,
 
 pp. 773–782 .
 
 Cited by: TABLE II ,
 2nd item .
 

 [173] 
 L. Crocker and J. Algina (1986) 
 
 Introduction to classical and modern test theory. .
 
 ERIC .
 
 Cited by: §VIII .
 

 [174] 
 W. J. Van der Linden and C. A. Glas (2010) 
 
 Elements of adaptive testing .
 
 Vol. 10 , Springer .
 
 Cited by: §VIII ,
 §VIII .
 

 [175] 
 A. P. Bradley (1997) 
 
 The use of the area under the roc curve in the evaluation of machine learning algorithms .
 
 Pattern recognition 30 ( 7 ), pp. 1145–1159 .
 
 Cited by: §VIII .
 

 [176] 
 M. Feng, N. Heffernan, and K. Koedinger (2009) 
 
 Addressing the assessment challenge with an online system that tutors as it assesses .
 
 User modeling and user-adapted interaction 19 , pp. 243–266 .
 
 Cited by: 1st item ,
 §VIII .
 

 [177] 
 H. Chang, H. Hsu, and K. Chen (2015) 
 
 Modeling exercise relationships in e-learning: a unified approach. .
 
 In EDM ,
 
 pp. 532–535 .
 
 Cited by: 2nd item ,
 §VIII .
 

 [178] 
 Y. Choi, Y. Lee, D. Shin, J. Cho, S. Park, S. Lee, J. Baek, C. Bae, B. Kim, and J. Heo (2020) 
 
 Ednet: a large-scale hierarchical dataset in education .
 
 In Artificial Intelligence in Education: 21st International Conference, AIED 2020, Ifrane, Morocco, July 6–10, 2020, Proceedings, Part II 21 ,
 
 pp. 69–73 .
 
 Cited by: 4th item ,
 §VIII .
 

 [179] 
 Z. Wang, A. Lamb, E. Saveliev, P. Cameron, Y. Zaykov, J. M. Hernández-Lobato, R. E. Turner, R. G. Baraniuk, C. Barton, S. P. Jones, S. Woodhead, and C. Zhang (2020) 
 
 Diagnostic questions: the neurips 2020 education challenge .
 
 arXiv preprint arXiv:2007.12061 .
 
 Cited by: 5th item ,
 §VIII .
 

 [180] 
 A. Srivastava, A. Rastogi, A. Rao, A. A. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta, A. Garriga-Alonso, et al. (2023) 
 
 Beyond the imitation game: quantifying and extrapolating the capabilities of language models .
 
 Transactions on machine learning research .
 
 Cited by: §VIII .
 

 [181] 
 E. Beeching, C. Fourrier, N. Habib, S. Han, N. Lambert, N. Rajani, O. Sanseviero, L. Tunstall, and T. Wolf (2023) 
 
 Open llm leaderboard (2023-2024) .
 
 Hugging Face .
 
 Note: https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard 
 
 Cited by: §VIII .
 

 [182] 
 P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar, et al. (2022) 
 
 Holistic evaluation of language models .
 
 arXiv preprint arXiv:2211.09110 .
 
 Cited by: §VIII ,
 §IX .
 

 [183] 
 X. Li, T. Zhang, Y. Dubois, R. Taori, I. Gulrajani, C. Guestrin, P. Liang, and T. B. Hashimoto (2023) 
 
 Alpacaeval: an automatic evaluator of instruction-following models .
 
 Cited by: §VIII .
 

 [184] 
 C. Xiao, L. Shi, A. Cristea, Z. Li, and Z. Pan (2022) 
 
 Fine-grained main ideas extraction and clustering of online course reviews .
 
 In International Conference on Artificial Intelligence in Education ,
 
 pp. 294–306 .
 
 Cited by: §IX .
 

 [185] 
 L. Shi, A. I. Cristea, and S. Hadzidedic (2014) 
 
 Multifaceted open social learner modelling .
 
 In Advances in Web-Based Learning–ICWL 2014: 13th International Conference, Tallinn, Estonia, August 14-17, 2014. Proceedings 13 ,
 
 pp. 32–42 .
 
 Cited by: §IX .
 

 [186] 
 X. L. Dong, S. Moon, Y. E. Xu, K. Malik, and Z. Yu (2023) 
 
 Towards next-generation intelligent assistants leveraging llm techniques .
 
 In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining ,
 
 pp. 5792–5793 .
 
 Cited by: §IX .
 

 [187] 
 Y. Chang, X. Wang, J. Wang, Y. Wu, K. Zhu, H. Chen, L. Yang, X. Yi, C. Wang, Y. Wang, et al. (2023) 
 
 A survey on evaluation of large language models .
 
 arXiv preprint arXiv:2307.03109 .
 
 Cited by: §IX .
 

 [188] 
 S. Xu, W. Hua, and Y. Zhang (2024) 
 
 OpenP5: an open-source platform for developing, training, and evaluating llm-based recommender systems .
 
 In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval ,
 
 pp. 386–394 .
 
 Cited by: §IX .
 

 [189] 
 L. Zhu, X. Huang, and J. Sang (2024) 
 
 How reliable is your simulator? analysis on the limitations of current llm-based user simulators for conversational recommendation .
 
 In Companion Proceedings of the ACM on Web Conference 2024 ,
 
 pp. 1726–1732 .
 
 Cited by: §IX .
 

 [190] 
 S. Bhandari, Y. Liu, and Z. A. Pardos (2023) 
 
 Evaluating chatgpt-generated textbook questions using irt .
 
 In Generative AI for Education Workshop (GAIED) at the Thirty-seventh Conference on Neural Information Processing Systems ,
 
 Cited by: §IX .
 

 [191] 
 A. Robstad and R. L. Sadun (2024) 
 
 DaTT-it: exploring the effect of combining generative ai-generated feedback with computerized adaptive testing .
 
 Master’s thesis , Norwegian University of Science and Technology (NTNU), Faculty of Information Technology and Electrical Engineering, Department of Computer Science , ( English ).
 
 Cited by: §IX .
 

 
 
 
 
 
 | 
 
 
 Yan Zhuang received the Ph.D. degree from the University of Science and Technology of China (USTC), in 2025. He is currently an Associate Professor with Nanjing University of Aeronautics and Astronautics. His main research interests include data mining and intelligent education systems. He has published more than 20 papers in top conferences and journals such as NeurIPS, ICML, ICLR, AAAI, and IEEE TPAMI. He received the Best Paper Runner-Up Award at CIKM 2023. 
 | 

 
 
 
 
 | 
 
 
 Qi Liu (Member, IEEE)
received the Ph.D. degree from the University of Science and Technology of China (USTC), in 2013. He is currently a Professor with USTC. His general research areas include data mining and knowledge discovery, and artificial intelligence. His research is supported by the National Science Fund for Excellent Young Scholars and the Youth Innovation Promotion Association of Chinese Academy of Sciences. He has published more than 100 papers in refereed journals and conference proceedings, such as TKDE, TOIS, TNNLS, NeurIPS, ICML, ICLR, and KDD. Dr. Liu is the recipient of the KDD 2018 Best Student Paper Award (Research) and the ICDM 2011 Best Research Paper Award. 
 | 

 
 
 
 
 | 
 
 
 Haoyang Bi 
received the B.E. degree in computer science and technology from University of
Science and Technology of China (USTC), Hefei, China, in 2019. He is currently a Ph.D. student in the School of Computer Science and Technology at University of Science and Technology of China (USTC), China. His research interests include active learning, Bayesian learning and meta-learning. 
 | 

 
 
 
 
 | 
 
 
 Zhenya Huang (Member, IEEE)
received the Ph.D. degree from the University of Science and Technology of China (USTC), in 2020. He is currently an Associate Professor with USTC. His main research interests include artificial intelligence, knowledge reasoning, and intelligent education. He has published more than 50 papers in refereed journals and conference proceedings, including TKDE, TOIS, TNNLS, AAAI, KDD, SIGIR, and ICDM. Dr. Huang has served regularly on the program committee of numerous conferences and is a reviewer for the leading academic journals. 
 | 

 
 
 
 
 | 
 
 
 Weizhe Huang 
received his Bachelor’s degree in computer science from University of Science and Technology of China (USTC) in 2022. He is currently pursuing a Master’s degree at USTC. His research interests include sequence modeling, computerized adaptive testing, and educational data mining. 
 | 

 
 
 
 
 | 
 
 
 Jiatong Li 
received his BS degree from University of Science and Technology of China (USTC). He is currently working toward the master degree in School of Artificial Intelligence and Data Science, USTC. His research interests include educational data mining, trustworthy AI and model evaluation. His works in educational data mining have been published in major conference in related fields such as KDD, WWW, etc. 
 | 

 
 
 
 
 | 
 
 
 Junhao Yu 
He is currently working toward the master degree at the University of Science and Technology of China. His main research interests include artificial intelligence, large language models, data mining, and adaptive testing. 
 | 

 
 
 
 
 | 
 
 
 Zirui Liu 
is master student in the University of Science and Technology of China (USTC). His main research interests include data mining and intelligent education. 
 | 

 
 
 
 
 | 
 
 
 Zirui Hu 
received his master’s degree from the University of Science and Technology of China (USTC). His research interests include fairness in recommender systems, causal inference, and intelligent education. He has published his work on fair learning in major conferences in these fields, such as DASFAA and KSEM, etc. 
 | 

 
 
 
 
 | 
 
 
 Yuting Hong 
received the Masters’ degree from the University of Science and Technology of China (USTC), in 2024. Her work in Computerized Adaptive Testing has been published in CIKM and received the Best Paper Runner-Up on CIKM 2023. 
 | 

 
 
 
 
 | 
 
 
 Zachary A. Pardos 
earned his PhD in Computer Science at Worcester Polytechnic Institute. He is an Associate Professor of Education at UC Berkeley studying adaptive learning and AI. His early scholarship focused on formative assessment using Knowledge Tracing, the predominant model used for estimating skill mastery in computer tutoring system contexts. His recent work designing Human-AI collaborations to pave pathways to and within higher education systems has been published in venues such as SIGCHI, AAAI, The Internet and Higher Education, and Science. 
 | 

 
 
 
 
 | 
 
 
 Haiping Ma 
received the BE degree from Anhui University, Hefei, China, in 2008, and the PhD degree
from the University of Science and Technology of China, Hefei, China, in 2013. She is currently an associate professor with the Institutes of Physical Science and Information Technology, Anhui University,Hefei, China. Her current research interests include data mining and multi-objective optimization methods and their applications. 
 | 

 
 
 
 
 | 
 
 
 Mengxiao Zhu (Member, IEEE) received the Ph.D. degree in industrial engineering and management sciences from Northwestern University in 2012. She has been a Distinguished Research Professor at the University of Science and Technology of China (USTC) since 2020. Before joining USTC, she worked as a Research Scientist in the Research and Development division at Educational Testing Service (ETS) for over seven years. She has been leading and involved in multiple NSFC, NSF, and NIH-funded projects in the past 20 years. 
 | 

 
 
 
 
 | 
 
 
 Shijin Wang 
received the Ph.D. degree from the Institute of Automation, Chinese Academy of Science. He is currently the vice president of IFLYTEK Co., Ltd. and the president of IFLYTEK AI Research (Central China). His research interests include speech and natural language processing. He has published more than 60 papers in refereed conferences such as ACL, KDD, and AAAI. He led the team that won more than ten championships in international technical evaluation such as Blizzard Challenge and CHiME. 
 | 

 
 
 
 
 | 
 
 
 Enhong Chen (Fellow, IEEE)
received the Ph.D. degree from the University of Science and Technology of China (USTC), in 1996. He is currently a Professor and the Vice Director of State Key Laboratory of Cognitive Intelligence. His research areas include data mining and machine learning, artificial intelligence. His research is supported by the National Science Foundation for Distinguished Young Scholars of China. He has published more than 200 papers in refereed conferences and journals, including TPAMI, TKDE, TNNLS, TOIS, ICML, NeurIPS, KDD, ICLR and AAAI. He is an associate editor of the IEEE TKDE, IEEE TSMCS, ACM TIST, WWWJ. Dr. Chen received the Best Application Paper Award on KDD 2008, the Best Research Paper Award on ICDM 2011, the Best Student Paper Award on KDD 2018 (Research), and the Best Student Paper Award on KDD 2024 (Research). 
 | 

 
 
 
 

## Comparison of Fisher Information and KL Information

 
 Fig. 7 illustrates the KL and Fisher information functions for two distinct questions. For θ \theta near θ 0 \theta_{0} , KL Information and Fisher information are always close. If we envision KL Information as a curve, Fisher information corresponds to its curvature (second derivative) at θ = θ 0 \theta=\theta_{0} . This suggests that Fisher information can be derived from KL Information, but the converse is not true.

 
 
 Fig. 7: Illustration of KL and Fisher information functions for two questions (Question 1: α = 1.7 , β = 1.9 , c = 0.1 \alpha=1.7,\beta=1.9,c=0.1 ; Question 2: α = 2.3 , β = 0.5 , c = 0.3 \alpha=2.3,\beta=0.5,c=0.3 ). Assuming the current proficiency estimate θ ^ t = 1 \hat{\theta}^{t}=1 . The KL information (left) for the given question represents an integral centered around θ ^ t \hat{\theta}^{t} , while the Fisher information (right) corresponds to the value at the specific point θ ^ t \hat{\theta}^{t} . 
 
 
 

## Analysis of Various Key Factors in Testing

 
 Table II showcases the underlying causes and advantages of different factors in CAT test control.

 
 
 TABLE II: Test Control: Key Factors in CAT Implementation 
 
 
 
 
 
 Factors 
 | 
 
 
 Category 
 | 
 
 
 Causes 
 | 
 
 
 Advantages 
 | 
 
 
 Pubs 
 | 

 
 
 
 Exposure Control 
 | 
 
 
 – 
 | 
 
 
 Unbalanced question usage 
 | 
 
 
 Mitigates overexposure; 
 Test security; 
 Comprehensive assessment 
 | 
 
 
 [ 135 , 136 ] 
 [ 139 , 141 ] 
 [ 16 , 86 ] 
 | 

 
 
 
 Fairness 
 | 
 
 
 Bias in Measurement 
 Models 
 | 
 
 
 Skewed training data;
 Underrepresentation of certain groups 
 | 
 
 
 Promotes equitable outcomes;
 Improves accuracy of proficiency estimation 
 | 
 
 
 [ 150 , 149 ] 
 
 [ 148 ] 
 | 

 
 | 
 
 
 Bias in question Bank 
 | 
 
 
 Unequal applicability; 
 
 Cultural or regional biases 
 | 
 
 
 Ensures content relevance; 
 Reduces disadvantage for certain groups 
 | 
 
 
 [ 151 , 152 ] 
 
 [ 153 , 154 ] 
 | 

 
 | 
 
 
 Bias in Selection 
 Algorithms 
 | 
 
 
 Algorithmic preferences 
 | 
 
 
 Reduces disadvantage for certain groups 
 | 
 
 
 [ 14 ] 
 | 

 
 | 
 
 
 Equating 
 | 
 
 
 Different selected questions across examinees 
 | 
 
 
 Score comparability; 
 Fairness across different tests 
 | 
 
 
 [ 156 , 158 ] 
 [ 159 , 160 ] 
 | 

 
 
 
 Robustness 
 | 
 
 
 Noise Resistance 
 | 
 
 
 Random variability;
 Guessing and slipping factors 
 | 
 
 
 Stabilizes estimation; 
 Improves reliability 
 | 
 
 
 [ 161 , 92 ] 
 | 

 
 | 
 
 
 Modeling Uncertainty 
 | 
 
 
 Uncertainty in response 
 | 
 
 
 Improves accuracy of proficiency estimation 
 | 
 
 
 [ 166 , 92 ] 
 | 

 
 
 
 Search Efficiency 
 | 
 
 
 – 
 | 
 
 
 Large question banks; 
 Brute-force search 
 | 
 
 
 Reduces search complexity 
 | 
 
 
 [ 168 , 172 ] 
 | 

 
 
 
 

## Comparison of Different Selection Algorithms

 
 Table III displays some representative methods of each category of selection algorithms and their AUC results on two different datasets. The comparison in this survey focuses on the results at the early testing stage (step=5) and the final testing stage (step=20). It is important to note that the results cannot be directly compared if experimental settings are not standardized. Despite this, the table as a whole reveals that data-driven methods (e.g., reinforcement learning, meta-learning methods) generally outperform statistical methods. This is because these methods can train and optimize selection algorithms from examinee large-scale response data, while statistical methods simply adhere to fixed functions for selecting questions. The latest subset selection methods do not require training but are remarkably effective. This is primarily because they attempt to explicitly approximate the objectives of CAT and provide theoretical guarantees on estimation errors. Furthermore, it is observed that considering factors within test control, such as robustness, can enhance accuracy.

 
 
 TABLE III: AUC results reported by different CAT methods 
 
 
 
 | 
 ASSISTments | 
 Eedi2020 | 

 
 
 Selection Algorithm Measurement Model | 
 IRT [ 46 ] | 
 NeuralCD [ 22 ] | 
 IRT [ 46 ] | 
 NeuralCD [ 22 ] | 

 
 AUC@5 | 
 AUC@20 | 
 AUC@5 | 
 AUC@20 | 
 AUC@5 | 
 AUC@20 | 
 AUC@5 | 
 AUC@20 | 

 
 Random | 
 
 
 
 67.68 [ 86 ] | 

 
 70.68 [ 21 ] | 

 
 65.86 [ 85 ] | 

 | 
 
 
 
 68.43 [ 86 ] | 

 
 72.61 [ 21 ] | 

 | 
 
 
 
 67.73 [ 86 ] | 

 
 71.19 [ 21 ] | 

 
 70.52 [ 100 ] | 

 | 
 
 
 
 69.70 [ 86 ] | 

 
 72.83 [ 21 ] | 

 | 
 
 
 
 68.38 [ 86 ] | 

 
 69.05 [ 21 ] | 

 | 
 
 
 
 71.98 [ 86 ] | 

 
 74.82 [ 21 ] | 

 | 
 
 
 
 68.45 [ 86 ] | 

 
 69.32 [ 21 ] | 

 
 73.67 [ 100 ] | 

 | 
 
 
 
 72.98 [ 86 ] | 

 
 74.99 [ 21 ] | 

 | 

 
 Statistical Algorithms | 
 Fisher Information [ 14 ] | 
 
 
 
 67.95 [ 86 ] | 

 
 71.33 [ 21 ] | 

 
 66.41 [ 85 ] | 

 | 
 
 
 
 69.26 [ 86 ] | 

 
 73.54 [ 21 ] | 

 | 
 – | 
 – | 
 
 
 
 68.92 [ 86 ] | 

 
 70.60 [ 21 ] | 

 | 
 
 
 
 72.66 [ 86 ] | 

 
 76.24 [ 21 ] | 

 | 
 – | 
 – | 

 
 KL Information [ 15 ] | 
 
 
 
 67.92 [ 86 ] | 

 
 71.38 [ 21 ] | 

 | 
 
 
 
 69.23 [ 86 ] | 

 
 73.57 [ 21 ] | 

 | 
 – | 
 – | 
 
 
 
 68.69 [ 86 ] | 

 
 69.79 [ 21 ] | 

 | 
 
 
 
 72.60 [ 86 ] | 

 
 75.73 [ 21 ] | 

 | 
 – | 
 – | 

 
 
 
 
 Fisher Information | 

 
 + Robust [ 92 ] | 

 | 
 – | 
 – | 
 – | 
 – | 
 68.93 [ 92 ] | 
 75.99 [ 92 ] | 
 – | 
 – | 

 
 
 
 
 KL Information | 

 
 + Robust [ 92 ] | 

 | 
 – | 
 – | 
 – | 
 – | 
 68.90 [ 92 ] | 
 76.03 [ 92 ] | 
 – | 
 – | 

 
 Active Learning | 
 MAAT [ 16 ] | 
 
 
 
 68.24 [ 86 ] | 

 
 71.54 [ 21 ] | 

 
 66.24 [ 85 ] | 

 | 
 
 
 
 69.7 [ 86 ] | 

 
 73.08 [ 21 ] | 

 | 
 
 
 
 67.96 [ 86 ] | 

 
 70.98 [ 21 ] | 

 
 70.85 [ 100 ] | 

 | 
 
 
 
 71.17 [ 86 ] | 

 
 72.27 [ 21 ] | 

 | 
 
 
 
 69.09 [ 86 ] | 

 
 70.32 [ 21 ] | 

 | 
 
 
 
 73.19 [ 86 ] | 

 
 74.46 [ 21 ] | 

 | 
 
 
 
 69.03 [ 86 ] | 

 
 70.12 [ 21 ] | 

 
 74.33 [ 100 ] | 

 | 
 
 
 
 73.75 [ 86 ] | 

 
 75.83 [ 21 ] | 

 | 

 
 
 
 
 MAAT | 

 
 + Robust [ 92 ] | 

 | 
 – | 
 – | 
 – | 
 – | 
 68.93 [ 92 ] | 
 76.09 [ 92 ] | 
 70.39 [ 92 ] | 
 76.63 [ 92 ] | 

 
 
 
 
 Reinforcement 
 
 Learning 
 | 
 GMOCAT [ 86 ] | 
 69.13 [ 86 ] | 
 71.91 [ 86 ] | 
 69.95 [ 86 ] | 
 72.95 [ 86 ] | 
 69.81 [ 86 ] | 
 74.19 [ 86 ] | 
 71.25 [ 86 ] | 
 75.76 [ 86 ] | 

 
 NCAT [ 24 ] | 
 
 
 
 68.67 [ 86 ] | 

 
 71.53 [ 21 ] | 

 | 
 
 
 
 71.06 [ 86 ] | 

 
 73.50 [ 21 ] | 

 | 
 
 
 
 69.28 [ 86 ] | 

 
 71.59 [ 21 ] | 

 
 72.53 [ 100 ] | 

 | 
 
 
 
 71.68 [ 86 ] | 

 
 73.59 [ 21 ] | 

 | 
 
 
 
 69.04 [ 86 ] | 

 
 72.11 [ 21 ] | 

 | 
 
 
 
 73.32 [ 86 ] | 

 
 76.66 [ 21 ] | 

 | 
 
 
 
 69.09 [ 86 ] | 

 
 74.10 [ 21 ] | 

 
 74.49 [ 100 ] | 

 | 
 
 
 
 74.55 [ 86 ] | 

 
 79.12 [ 21 ] | 

 | 

 
 
 
 
 Meta Learning 
 
 Algorithms 
 | 
 BOBCAT [ 26 ] | 
 
 
 
 68.65 [ 86 ] | 

 
 71.68 [ 21 ] | 

 
 66.41 [ 85 ] | 

 | 
 
 
 
 70.97 [ 86 ] | 

 
 73.39 [ 21 ] | 

 | 
 
 
 
 69.50 [ 86 ] | 

 
 71.45 [ 21 ] | 

 
 71.98 [ 100 ] | 

 | 
 
 
 
 71.80 [ 86 ] | 

 
 72.84 [ 21 ] | 

 | 
 
 
 
 68.94 [ 86 ] | 

 
 74.42 [ 21 ] | 

 | 
 
 
 
 73.24 [ 86 ] | 

 
 76.58 [ 21 ] | 

 | 
 
 
 
 69.17 [ 86 ] | 

 
 76.00 [ 21 ] | 

 
 75.12 [ 100 ] | 

 | 
 
 
 
 74.51 [ 86 ] | 

 
 79.00 [ 21 ] | 

 | 

 
 DL-CAT [ 85 ] | 
 66.68 [ 85 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 SACAT [ 100 ] | 
 – | 
 – | 
 75.24 [ 100 ] | 
 – | 
 – | 
 – | 
 75.48 [ 100 ] | 
 – | 

 
 
 
 
 Subset Selection | 

 
 Algorithms | 

 | 
 BECAT [ 21 ] | 
 71.44 [ 21 ] | 
 73.61 [ 21 ] | 
 71.60 [ 21 ] | 
 73.70 [ 21 ] | 
 73.15 [ 21 ] | 
 76.82 [ 21 ] | 
 76.30 [ 21 ] | 
 79.36 [ 21 ] | 

 
 
 
 

## Introduction to Representative Datasets

 
 The following are introductions to several commonly used datasets, and more datasets can be found at our EduData GitHub link: https://github.com/bigdata-ustc/EduData 

 
 
 
 • 
 
 ASSISTments [ 176 ] , established in 2004, is an online tutoring platform in the United States that offers examinees both assessments and instructional support. To date, the ASSISTments team has released four public datasets 3 3 
 3 
 
 
 
 https://sites.google.com/site/assistmentsdata/datasets/ : ASSISTments2009, ASSISTments2012, ASSISTments2015, and ASSISTments2017. These datasets are response data and mostly collected from mathematics in middle school. They also include valuable side information, such as attempt count (the number of tries an examinee has made), ms first response (the time it takes for an examinee’s first response), problem type, and average confidence.

 

 • 
 
 Junyi Dataset [ 177 ] includes logs and exercise data from Junyi Academy, a Chinese online learning platform launched in 2012 using Khan Academy’s open-source code. It features a detailed question hierarchy and relationships, labeled by experts.

 

 • 
 
 MOOCCube 4 4 
 4 
 
 
 
 https://www.biendata.xyz/competition/chaindream_mooccube_task2/ , Massive Open Online Courses (MOOCs) are among the most prevalent platforms for online learning. This dataset collects examinees’ responses to questions related to various computer science knowledge concepts. Additionally, the dataset includes the text of the problems, which can be used to enhance the performance of question selection, proficiency estimation, question characteristics analysis, etc.

 

 • 
 
 EdNet Dataset [ 178 ] is a large collection of examinee learning records from the AI tutoring system Santa 5 5 
 5 
 
 
 
 https://github.com/riiid/ednet , which is used for English language learning in South Korea. It focuses on examinees preparing for the eTOEIC (Test of English for International Communication) Listening and Reading Test, with over 131 million learning records from approximately 784,000 examinees.

 

 • 
 
 Eedi2020 Dataset [ 179 ] , released for the NeurIPS 2020 Education Challenge, contains over 17 million records of examinees’ responses to mathematics multiple-choice questions on the Eedi platform 6 6 
 6 
 
 
 
 https://eedi.com/projects/neurips-education-challenge . It includes detailed information on examinees’ choices, demographics, and containment relationships of knowledge concepts, as well as associated quiz and curriculum metadata. This extensive dataset enables in-depth analysis of examinee behaviors and the development of personalized tools.

 

 
 
 
 

## Systematic Literature Review Protocol

 
 To improve the transparency and reproducibility of this survey, we followed a lightweight SLR-style protocol for collecting and screening the literature. We searched major scholarly databases and digital libraries (e.g., Google Scholar, IEEE Xplore, ACM Digital Library, and arXiv) using keyword combinations related to computerized adaptive testing and psychometrics (e.g., “computerized adaptive testing”, “CAT”, “item response theory/IRT”, “exposure control”, “content balancing”, “online calibration”, “multidimensional IRT”) as well as recent extensions to AI/LLM evaluation (e.g., “adaptive evaluation”, “LLM benchmarking”, “agent-based assessment”). We focused primarily on peer-reviewed papers and widely used technical reports within the period 2000–2025, while allowing earlier seminal works when necessary for completeness. We applied inclusion criteria requiring clear methodological relevance to CAT/IRT (or their use in AI model evaluation). The screening was conducted in two stages: an initial title/abstract filtering followed by full-text review for highly relevant candidates. The selected studies were then organized into the taxonomy and sections presented (e.g., Figure 2, Table 2 and 3).”