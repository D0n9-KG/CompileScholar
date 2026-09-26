The Pursuit of Fairness in Artificial Intelligence Models: A Survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2403.17333v1 [cs.AI] 26 Mar 2024 
 
 

# The Pursuit of Fairness in Artificial Intelligence Models: A Survey

 DOI:  XXXXXXX.XXXXXXX CCS:  Computing methodologies Artificial intelligence CCS:  Computing methodologies Machine learning 
 
 
 Tahsin Alamgir Kheya
 
 email: s224091662@deakin.edu.au 
 
 Affiliation:  Deakin University , Waurn Ponds , Victoria , Australia , 3216 
 
 , 
 Mohamed Reda Bouadjenek
 
 email: reda.bouadjenek@deakin.edu.au 
 
 Affiliation:  Deakin University , Waurn Ponds , Victoria , Australia 
 
 and 
 Sunil Aryal
 
 email: sunil.aryal@deakin.edu.au 
 
 Affiliation:  Deakin University , Waurn Ponds , Victoria , Australia 
 
 2024 

 Abstract. 
 
 Artificial Intelligence (AI) models are now being utilized in all facets of our lives such as healthcare, education and employment. Since they are used in numerous sensitive environments and make decisions that can be life altering, potential biased outcomes are a pressing matter. Developers should ensure that such models don’t manifest any unexpected discriminatory practices like partiality for certain genders, ethnicities or disabled people. With the ubiquitous dissemination of AI systems, researchers and practitioners are becoming more aware of unfair models and are bound to mitigate bias in them. Significant research has been conducted in addressing such issues to ensure models don’t intentionally or unintentionally perpetuate bias. This survey offers a synopsis of the different ways researchers have promoted fairness in AI systems. We explore the different definitions of fairness existing in the current literature. We create a comprehensive taxonomy by categorizing different types of bias and investigate cases of biased AI in different application domains. A thorough study is conducted of the approaches and techniques employed by researchers to mitigate bias in AI models. Moreover, we also delve into the impact of biased models on user experience and the ethical considerations to contemplate when developing and deploying such models. We hope this survey helps researchers and practitioners understand the intricate details of fairness and bias in AI systems. By sharing this thorough survey, we aim to promote additional discourse in the domain of equitable and responsible AI.

 
 
 
 Keywords:  Fair AI, Fairness in Artificial Intelligence, Fair Machine Learning models, Bias in AI, Bias in Artificial Intelligent, Biased Models
 
 

## 1. Introduction

 
 The use of automated systems has rapidly advanced across various domains, influencing everything from hiring employees to recommendation systems. AI systems are embedded in our day-to-day activities and influence our lives greatly, especially when used to make life-altering decisions. These models have great potential, as they can integrate tons of data and perform very complex computations more effectively and faster than humans. Amid AI’s potential, however, concerns arise regarding the fairness and bias in these systems. Since these systems are being used in sectors like healthcare, finance, and criminal justice to make prominent decisions for individuals, ensuring fairness in these models is critical.
 
 In recent years a number of cases of AI bias has been exposed and the major effect it has on individuals and communities is inevitable. For instance, in the USA an algorithm used to find the recidivism score for sentencing was found to be biased towards Black defendants ( Mattu et al., 2016 ) . Google Bard was seen to depict gender stereotypes by stating boys want to achieve goals and make a difference in life while girls want love and affection ( Fowler, 2023 ) . These are just two examples, but numerous concerns like these have led to growing interest in developing and deploying fair AI models. Fig 1 shows the number of papers published in this field for the last seven years. Over the years the volume of papers published in this domain has steadily increased. By 2021, the number of papers surged over 1000. The steady increase resulted in numbers close to a whopping 2000 papers published last year.
The graph underscores the significance of these topics in the research community over the years.
 
 When evaluating models’ fairness more than one definition of fairness has been used. This survey explores all the different fairness criteria discussed in the literature. Several researchers have been working extensively to address fairness issues in automated models. With the broad domain of fair AI, researchers have put forward multiple strategies to address and mitigate bias in them. It is also important to realize that certain strategies only work on certain types of bias. This paper thoroughly describes the different types of biases and all the common approaches used to mitigate these biases. Moreover, this survey covers details on the causes of unfairness, different cases of biases within different sectors including but not limited to healthcare, education and finance. Working towards making AI models fair can also enhance user experience. In this paper, we discuss the impact of biased models on users and ethical guidelines that should be followed to ensure users’ trust. At the end of the paper, we mention the challenges and limitations of the current literature. Overall the aim of the paper is to shed light on the existing work done on bias and fairness in the context of AI models. We hope this paper will provide researchers and practitioners with enriched perspectives in this field and encourage them to decide on their research direction and develop innovative ideas to mitigate unintended consequences.

 
 
 Figure 1. Number of papers published in this topic over the years. Data acquisition process to plot this graph is provided in Section A.2 . 
 
 
 

## 2. Related Survey

 
 In recent years bias and fairness in the context of machine learning has been a hot topic. Numerous surveys have been written in this domain. These recent surveys ( Mehrabi et al., 2021 ; Ferrara, 2023 ; Pessach and Shmueli, 2022 ; Caton and Haas, 2023 ; Tang et al., 2023 ; Ntoutsi et al., 2020 ; Chen et al., 2023b ) are thorough and provide insight into the cause of bias, mitigating them and promoting fairness in a more general context. There are several papers that explore fairness in a specific domain which include recommender systems ( Deldjoo et al., 2023 ; Chen et al., 2023a ) , vision language models ( Lee et al., 2023 ; Parraga et al., 2023 ) , healthcare ( Ueda et al., 2024 ; Fletcher et al., 2021 ; Timmons et al., 2023 ) , finance ( Pavón Pérez, 2022 ) and NLP ( Blodgett et al., 2020 ; Bansal, 2022 ; Garrido-Muñoz  et al., 2021 ) . Our survey incorporates recent research which includes more updated definitions of bias and fairness, mitigation strategies and causes of unfairness in real life cases. Moreover almost none of the previous surveys delves into details about how user experience was impacted because of bias in ML models and how this phenomenon has affected different sectors like education, recruitment etc. This paper thoroughly explores these neglected sub-topics.
Table 1 , showcases the subjects that were covered by other papers (akin to this paper) and some that were overlooked. The table is divided into 2 sections. The first few rows describe papers that capture the broad domain of fairness and bias in AI, and the second part describes surveys that cover specific domains like healthcare, NLP, and more.

 
 
 Table 1. A comparison of topics covered and overlooked by the recent surveys resembling this paper 
 
 
 
 Year | 
 Paper | 
 Fairness | 
 Types of | 
 Causes of | 
 Cases of | 
 Biased Models within | 
 Mitigation | 
 The Impact Bias has on | 
 Ethical | 

 
 | 
 | 
 Definition | 
 Bias | 
 Unfairness | 
 Bias | 
 Different Sectors | 
 Strategies | 
 User Experience | 
 Considerations | 

 
 | 
 This Paper | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 

 
 2020 | 
 ( Ntoutsi et al., 2020 ) | 
 ✓ | 
 x | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2021 | 
 ( Mehrabi et al., 2021 ) | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2022 | 
 ( Pessach and Shmueli, 2022 ) | 
 ✓ | 
 x | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2023 | 
 ( Caton and Haas, 2023 ) | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2023 | 
 ( Tang et al., 2023 ) | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2023 | 
 ( Ferrara, 2023 ) | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 ✓ | 
 x | 

 
 2023 | 
 ( Chen et al., 2023b ) | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2023 | 
 ( Siddique et al., 2024 ) | 
 x | 
 ✓ | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2023 | 
 ( Kaur et al., 2022 ) | 
 x | 
 ✓ | 
 x | 
 x | 
 x | 
 ✓ | 
 x | 
 ✓ | 

 
 Specialized surveys | 

 
 2020 | 
 ( Blodgett et al., 2020 ) | 
 x | 
 x | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2021 | 
 ( Fletcher et al., 2021 ) | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2021 | 
 ( Kordzadeh and Ghasemaghaei, 2022 ) | 
 x | 
 ✓ | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2021 | 
 ( Kaur et al., 2021 ) | 
 x | 
 ✓ | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 ✓ | 

 
 2021 | 
 ( Perrier, 2021 ) | 
 ✓ | 
 x | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2021 | 
 ( Garrido-Muñoz  et al., 2021 ) | 
 x | 
 ✓ | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2022 | 
 ( Lee et al., 2023 ) | 
 ✓ | 
 x | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2022 | 
 ( Bansal, 2022 ) | 
 ✓ | 
 ✓ | 
 x | 
 x | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2023 | 
 ( Timmons et al., 2023 ) | 
 x | 
 ✓ | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2023 | 
 ( Correa et al., 2022 ) | 
 x | 
 ✓ | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2023 | 
 ( Deldjoo et al., 2023 ) | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2023 | 
 ( Parraga et al., 2023 ) | 
 ✓ | 
 x | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 x | 

 
 2024 | 
 ( Ueda et al., 2024 ) | 
 x | 
 ✓ | 
 ✓ | 
 ✓ | 
 x | 
 ✓ | 
 x | 
 ✓ | 

 
 
 
 

## 3. Conceptualizing Fairness and Bias in ML

 
 In machine learning, fairness and bias are intertwined concepts that follow the same aspect: how models’ predictions can favor a particular group of people. Bias in the model will lead to unfairness; to get a fair model, we must mitigate bias. In simpler terms, bias is the issue, and fairness is the solution. Although fairness is stated to be the solution, it is important to note that achieving absolute fairness is a very challenging task, primarily because of the different fairness criteria. So, till now, there has been no single solution that mitigate all types of bias and makes a model absolutely fair.

 
 {forest} 
 Figure 2. Proposed taxonomy of fairness in the machine learning context 
 
 

### 3.1. Fairness in Machine Learning

 
 In simple terms, fairness in machine learning refers to models treating individuals and groups in an ethical manner while making predictions. In recent years, extensive research has been conducted on producing fair models based on different definitions of fairness. Figure 2 presents a taxonomy of fairness definitions utilized in previous research. It is compiled based on the concept of fairness discussed in existing literature such as ( Finocchiaro et al., 2021 ; Saxena et al., 2019 ; Calegari et al., 2023 ; Gohar and Cheng, 2023 ) .

 
 

#### 3.1.1. Group fairness

 
 Group Fairness, in the context of machine learning, describes the phenomena of the model treating different groups equally. For example, the ML model should treat people of different genders impartially. The concept of group fairness can be subdivided into various definitions.

 
 
 3.1.1.1 Demographic Parity 

 
 Demographic parity ensures that the predicted outcome Y ^ \hat{Y} of an ML model is independent of sensitive attribute S S ( Grari et al., 2020 ) . For example, a model’s prediction of a potential candidate to be hired for a position should not depend on the candidate’s gender. For binary classification with a binary sensitive attribute S S , demographic parity can be formalized as ( Hardt et al., 2016 ) :

 

 
 | 
 P ⁡ ( Y ^ = 1 | S = 0 ) = P ⁡ ( Y ^ = 1 | S = 1 ) P(\hat{Y}=1|S=0)=P(\hat{Y}=1|S=1) | 
 | 
 

 .

 
 
 
 3.1.1.2 Conditional Statistical Parity 

 
 Conditional statistical parity, also known as conditional demographic parity, requires the outcome for different sensitive groups to be the same, even after adding extra features ( Corbett-Davies et al., 2017 ) . For binary classification with binary sensitive attribute S S , and F F which is another feature (where f f is the value of this feature) conditional statistical parity can be formalized as:

 

 
 | 
 P ⁡ ( Y ^ = 1 | S = 0 , F = f ) = P ⁡ ( Y ^ = 1 | S = 1 , F = f ) P(\hat{Y}=1|S=0,F=f)=P(\hat{Y}=1|S=1,F=f) | 
 | 
 

 For example, for a model to decide if a student should be admitted to a school, the additional features considered could be GPA and admission test scores. Conditional statistical parity is satisfied as long as a similar number of male and female students are admitted for any combination of academic performance.

 
 
 
 

#### 3.1.2. Individual fairness

 
 Individual fairness focuses on ensuring that similar individuals receive similar predictions from a ML model, regardless of their membership in protected groups. In other words, this definition ensures individuals are characterized by their individual traits and not by any group stereotypes. Fairness through awareness and fairness through unawareness both fall under the umbrella of individual fairness because they are related to how individuals are treated by the ML model on a one-to-one basis.

 
 
 3.1.2.1 Fairness through Unawareness 

 
 A model is said to follow this definition of fairness as long as it doesn’t use any sensitive attributes when making decisions ( Kusner et al., 2018 ) . The equation

 

 
 | 
 Y : X → Y ^ Y:X\to\hat{Y} | 
 | 
 

 represents the function of the model (which takes input X and predicts outcome Y ^ \hat{Y} ) ( Kusner et al., 2018 ) . The main point here is that this mapping should exclude any sensitive attribute S S . Although this definition is very simple, other features used to train the model can contain discriminatory information ( Kusner et al., 2018 ; Grari et al., 2020 ; Calegari et al., 2023 ) . This can lead the model to infer attributes like gender, ethnic background, etc., from these features, leading to unfair predictions.

 
 
 
 3.1.2.2 Fairness through Awareness 

 
 The concept of fairness through awareness is a bit more sophisticated. This approach explicitly considers the sensitive attributes when the model is trained. Here k k is a metric to compute the similarity of candidates and M M is a model/function to predict their selection probabilities. This fairness criterion will hold if for any two candidates a a and b b , the difference in the distribution assigned to them (denoted by D(Ma,Mb)) is less than or equal to their similarity (denoted by k(a,b), i.e.,

 

 
 | 
 D ⁡ ( M ​ a , M ​ b ≤ k ⁡ ( a , b ) CLOSE D(Ma,Mb\leq k(a,b) | 
 | 
 

 
 
 
 

#### 3.1.3. Separation Metrics

 
 Separation metrics are a set of statistical criteria that try to enforce fairness by evaluating the model. The model is assessed to check the extent to which it separates the outcomes for different classes. There are several separation metrics, which are discussed in the subsections below.

 
 
 3.1.3.1 Predictive Equality 

 
 A model is said to satisfy this definition of fairness if the FPR (False Positive Rate) is equal for both protected group and unprotected group ( Verma and Rubin, 2018 ; Corbett-Davies et al., 2017 ) . For example, the probability of an individual with an actual bad credit score being incorrectly given a good credit score should be equal in different subgroups of a sensitive attribute ( Verma and Rubin, 2018 ) . With Y ^ \hat{Y} representing the prediction, S ∈ { 0 , 1 } S\in\{0,1\} a sensitive attribute and Y Y the actual outcome, this definition can be formalized as:

 

 
 | 
 P ⁡ ( Y ^ = 1 | S = 0 , Y = 0 ) = P ⁡ ( Y ^ = 1 | S = 1 , Y = 0 ) P(\hat{Y}=1|S=0,Y=0)=P(\hat{Y}=1|S=1,Y=0) | 
 | 
 

 
 
 
 3.1.3.2 Equal Opportunity 

 
 This concept of fairness ensures that individuals from different groups have an equal chance of receiving a positive outcome. In simple terms, this concept ensures individuals who obtain the "advantaged" outcome (e.g. successful candidate to be hired) have an equal chance of getting this prediction, regardless of any protected attribute ( Hardt et al., 2016 ) . This definition can be formalized for a binary classifier as ( Hardt et al., 2016 ) :

 

 
 | 
 P ⁡ ( Y ^ = 1 | S = 0 , Y = 1 ) = P ⁡ ( Y ^ = 1 | S = 1 , Y = 1 ) P(\hat{Y}=1|S=0,Y=1)=P(\hat{Y}=1|S=1,Y=1) | 
 | 
 

 It is quite similar to predictive equality but the focus for this definition is the true positive rate balance, whereas for predictive equality it is the false positive rate balance.

 
 
 
 3.1.3.3 Balance for the Negative Class 

 
 This concept of fairness ensures that the predicted scores assigned by the model to individuals belonging to the negative class are the same for both protected and unprotected groups ( Verma and Rubin, 2018 ) . With an average predicted probability score of Z, binary sensitive attribute S ∈ { 0 , 1 } \in\{0,1\} and predicted outcome Y ^ \hat{Y} this definition can be formalized as:

 

 
 | 
 P ⁡ ( Z | Y ^ = 0 , S = 1 ) = P ⁡ ( Z | Y ^ = 0 , S = 0 ) P(Z|\hat{Y}=0,S=1)=P(Z|\hat{Y}=0,S=0) | 
 | 
 

 For example, for this definition to be satisfied, a model used to hire teachers should provide the same scores of not being hired for the candidates, if deemed not suitable, regardless of the race of the individuals.

 
 
 
 3.1.3.4 Balance for the Positive Class 

 
 This concept tries to ensure that the same predicted scores are assigned by the model to individuals belonging to the positive class for both protected and unprotected groups ( Verma and Rubin, 2018 ) . With a predicted probability score Z, binary sensitive attribute S ∈ { 0 , 1 } \in\{0,1\} and predicted outcome Y ^ \hat{Y} this definition can be formalized as:

 

 
 | 
 P ⁡ ( Z | Y ^ = 1 , S = 1 ) = P ⁡ ( Z | Y ^ = 1 , S = 0 ) P(Z|\hat{Y}=1,S=1)=P(Z|\hat{Y}=1,S=0) | 
 | 
 

 Let’s consider a similar example to the last definition (balance for negative class). For this notion of fairness to be met, a model used to hire teachers should provide the same scores of being hired for the candidates, if deemed suitable regardless of the race of the candidate.

 
 
 
 3.1.3.5 Equalized Odds
 

 
 This concept of fairness holds if the predictor Y ^ \hat{Y} and sensitive attribute S S are independent given Y Y ( Hardt et al., 2016 ) . Y ^ \hat{Y} is allowed to depend on S S , when genuinely relevant to the outcome. This metric also ensures that the true positive rates and false positive rates are equal for different groups of individuals ( Verma and Rubin, 2018 ) . This promotes fairness whilst still allowing the model to leverage useful insights by not omitting sensitive attributes (only when appropriate). According to ( Hardt et al., 2016 ) , this definition can be formalized as:

 

 
 | 
 P ⁡ ( Y ^ = 1 | S = 0 , Y = y ) = P ⁡ ( Y ^ = 1 | S = 1 , Y = y ) , y ∈ { 0 , 1 } P(\hat{Y}=1|S=0,Y=y)=P(\hat{Y}=1|S=1,Y=y),\quad y\in\{0,1\} | 
 | 
 

 Let’s consider a model that suggests restaurants for you to eat. The sensitive attribute S S in this case is the religion of users. When training, the model is allowed to consider S S , but only if it is relevant to the true outcome. It might learn that individuals following religion R R tend to like fish and other sea foods. However, it cannot simply suggest just pescatarian options to individuals who follow religion R R . This is to make sure that the model bases its predictions on appropriate features like dietary restrictions, price range, location, etc.

 
 
 
 

#### 3.1.4. Intersectional Fairness

 
 Intersectional fairness acknowledges that individuals can encounter some form of discrimination because of their overlapping identities. This concept goes beyond traditional fairness notions and considers more than individual characteristics. According to ( Gohar and Cheng, 2023 ) , intersectional identities can intensify unfairness that’s not even present in constituent groups (like Black woman vs Black vs woman). Readers interested in the realm of intersectional fairness are directed to review ( Gohar and Cheng, 2023 ; Foulds and Pan, 2018 ) .

 
 
 

#### 3.1.5. Treatment Equality

 
 This concept of fairness aims to achieve equal proportions of false negatives to false positives for both unprotected and protected groups ( Berk et al., 2017b ) . This definition can be formalized as:

 

 
 | 
 F ​ N 1 F ​ P 1 = F ​ N 2 F ​ P 2 \frac{FN_{1}}{FP_{1}}=\frac{FN_{2}}{FP_{2}} | 
 | 
 

 where subscripts 1 and 2 represent protected and unprotected groups, respectively.

 
 
 

#### 3.1.6. Sufficiency Metrics

 
 These metrics ensure that a model is equally calibrated to make fair decisions for different sensitive groups, like ethnicity, religion, age etc. This idea can be subdivided into three definitions. These are discussed in the next sections.

 
 
 3.1.6.1 Equal Calibration 

 
 This concept of fairness ensures that for a given probability score Z Z , people in both protected and unprotected categories should possess an equal probability of being in the positive class ( Chouldechova, 2016 ) . This definition is formalized by ( Chouldechova, 2016 ) as:

 

 
 | 
 P ⁡ ( Y ^ = 1 | Z = z , S = 0 ) = P ⁡ ( Y ^ = 1 | Z = z , S = 1 ) P(\hat{Y}=1|Z=z,S=0)=P(\hat{Y}=1|Z=z,S=1) | 
 | 
 

 where, Y ^ \hat{Y} is the predicted outcome, Z Z is the score, and S S is the sensitive attribute. This definition is pretty similar to predictive parity (refer to 3.1.6.2 ) since they ensure accuracy for both groups but equal calibration applies beyond binary scores ( Chouldechova, 2016 ) . For instance, this definition holds if, for any given score z z , a model used to hire candidates, predicts the same chance of actually getting hired for different genders.

 
 
 
 3.1.6.2 Predictive Parity 

 
 This concept of fairness holds if the PPVs (Positive Predictive Value) for protected and unprotected groups are equal ( Verma and Rubin, 2018 ; Chouldechova, 2016 ) . This means an individual who was predicted to get a positive outcome should actually get a positive outcome. As stated by the authors in ( Verma and Rubin, 2018 ) , this definition can be formalized as

 

 
 | 
 P ⁡ ( Y = 1 | Y ^ = 1 , S = 0 ) = P ⁡ ( Y = 1 | Y ^ = 1 , S = 1 ) P(Y=1|\hat{Y}=1,S=0)=P(Y=1|\hat{Y}=1,S=1) | 
 | 
 

 where Y ^ \hat{Y} is the predicted outcome, Y Y is the true outcome and S S is the sensitive attribute. Let’s think about a model which is used to determine if a person will repay a loan. For this definition to be satisfied, the probability of being in a positive group (pay back loan) should be the same as their likelihood of actually paying back the loan.

 
 
 
 3.1.6.3 Conditional Use Accuracy Equality 

 
 This definition of fairness ensures that PPVs (Positive Predicted values) and NPVs (Negative Predicted Values) are the same regardless of what sensitive group an individual belongs to ( Verma and Rubin, 2018 ; Chouldechova, 2016 ) . Here, NPV describes the probability of an individual being predicted with a negative outcome actually being in the negative class. A formal definition of NPV is given as:

 

 
 | 
 N ​ P ​ V = T ​ r ​ u ​ e ​ N ​ e ​ g ​ a ​ t ​ i ​ v ​ e ​ s T ​ r ​ u ​ e ​ N ​ e ​ g ​ a ​ t ​ i ​ v ​ e ​ s + F ​ a ​ l ​ s ​ e ​ N ​ e ​ g ​ a ​ t ​ i ​ v ​ e ​ s NPV=\frac{TrueNegatives}{TrueNegatives+FalseNegatives} | 
 | 
 

 Here, PPV describes the probability of an individual being predicted with a positive outcome actually being in the positive class.
A formal definition of PPV is given as:

 

 
 | 
 P ​ P ​ V = T ​ r ​ u ​ e ​ P ​ o ​ s ​ i ​ t ​ i ​ v ​ e ​ s T ​ r ​ u ​ e ​ P ​ o ​ s ​ i ​ t ​ i ​ v ​ e ​ s + F ​ a ​ l ​ s ​ e ​ P ​ o ​ s ​ i ​ t ​ i ​ v ​ e ​ s PPV=\frac{TruePositives}{TruePositives+FalsePositives} | 
 | 
 

 Let’s consider a model that is used to decide if a person will repay a loan taken. For this definition to hold:
 
 a. Throughout all sensitive groups who are predicted to be in the negative group (doesn’t pay loan), the actual rate of not paying back the loan is the same.
 
 b. Throughout all sensitive groups who are predicted to be in the positive group (pay back loan), the actual rate of paying the loan is the same.
 
 The authors in ( Verma and Rubin, 2018 ) formalizes this definition as :

 

 
 | 
 ( P ⁡ ( Y = 1 | Y ^ = 1 , S = 0 ) = P ⁡ ( Y = 1 | Y ^ = 1 , S = 1 ) ) ∧ ( P ⁡ ( Y = 0 | Y ^ = 0 , S = 0 ) = P ⁡ ( Y = 0 | Y ^ = 0 , S = 1 ) ) (P(Y=1|\hat{Y}=1,S=0)=P(Y=1|\hat{Y}=1,S=1))\land(P(Y=0|\hat{Y}=0,S=0)=P(Y=0|\hat{Y}=0,S=1)) | 
 | 
 

 where Y ^ \hat{Y} is the predicted outcome, Y Y is the actual outcome and S S is the sensitive attribute.

 
 
 
 

#### 3.1.7. Causal-based Fairness

 
 This concept involves using additional knowledge, like insights from experts, to identify the causal structure of a particular case ( Calegari et al., 2023 ) . For example, exploring hypothetical situations and asking questions like "what would happen if an individual had a different race" ( Calegari et al., 2023 ) . This fairness concept is broken down into two further sub-categories discussed next.

 
 
 3.1.7.1 Counterfactual Fairness 

 
 For a model to be counter-factually fair, it needs to have the same predictions for individuals having the same relevant features even if the protected attributes are different. As stated by ( Kusner et al., 2018 ) , this definition can be formalized as:

 

 
 | 
 P ⁡ ( Y ^ S ← s ​ ( U ) = y | X = x , S = s ) = P ⁡ ( Y ^ S ← s ′ ​ ( U ) = y | X = x , S = s ) P(\hat{Y}_{S\leftarrow s}(U)=y|X=x,S=s)=P(\hat{Y}_{S\leftarrow s^{\prime}}(U)=y|X=x,S=s) | 
 | 
 

 where S S is the protected attribute, U U is the set of latent background variables, X X is the remaining attributes (which is under context X = x X=x and S = s S=s ). Here Y ^ S ← s ​ ( U ) \hat{Y}_{S\leftarrow s}(U) represents the counterfactual variable Y ^ \hat{Y} , when S is set to s by an external intervention ( Piccininni, 2022 ) .

 
 
 
 3.1.7.2 Unresolved Discrimination 

 
 This kind of discrimination can arise when a sensitive attribute unfairly impacts the predicted outcome. In a causal graph, variable V can exhibit unresolved discrimination if a directed path exists from S (a sensitive attribute) to V that is not blocked by a resolving variable ( Kilbertus et al., 2018 ) .

 
 
 Figure 3. Graph that exhibits unresolved discrimination. 
 
 
 For this to be true V itself should be a non-resolving variable. A resolving variable can be described as a variable that intervenes between a sensitive attribute and the predicted outcome with the intentions to justify any observed discrimination.

 
 
 Figure 3 shows a causal graph where race (S) is directly impacting the decision of the housing application (A), that cannot be attributed by any resolving variable. Housing choice isn’t a resolving variable, and since it is directly impacted by the sensitive attribute S it is causing an unresolved discrimination.

 
 
 
 
 

### 3.2. Bias in Machine learning

 
 Figure 4. Proposed taxonomy of observed biases in the machine learning pipeline 
 
 
 The word ‘bias’ is derived from the French word ‘biais’, which means slope. Later, this word’s meaning was expanded in English and is now used to refer to the inclination towards supporting or opposing a certain person or group in an unfair way. In Machine Learning (ML), the same concept is applicable. This section discusses different types of biases that can arise in the ML pipeline. Figure 4 shows the various biases that arise at every step of the ML pipeline. The list is compiled from existing research works and includes concepts mentioned in ( Fahse et al., 2021 ; Mehrabi et al., 2021 ; Gu and Oelke, 2019 ; Hellström et al., 2020 ; Suresh and Guttag, 2021 ) . The various types of biases are organized in 3 sections, which include data-driven, human and model bias.

 
 

#### 3.2.1. Data-Driven Bias

 
 This section describes biases that arise in the outcomes of models that learn patterns from the training data.

 
 
 3.2.1.1 Measurement Bias 

 
 Measurement bias is introduced when subjective choices are made for the model design ( Fahse et al., 2021 ) . This includes the act of selecting, gathering or computing features and annotations to be used in a prediction problem ( Suresh and Guttag, 2021 ) . One case of this kind of bias occurred when an algorithm was used to estimate grades of students who couldn’t sit for proper exams during COVID-19. This algorithm used historical school performance as a proxy to predict the abilities of the current students. It introduced measurement bias that unfairly affected students’ grades, particularly those with lower historical performance ( Denes, 2023 ) .

 
 
 
 3.2.1.2 Representation Bias 

 
 Representation bias can occur during the data collection or sampling phase when the probability distribution of the training samples is not the same as the true underlying distribution ( Fahse et al., 2021 ) . An example of this kind of bias is observed in the open-source image data set ImageNet. Shankar et al. ( Shankar et al., 2017 ) shows how this data set doesn’t have enough geographically diverse images that can have a broad representation across the changing world we live in.

 
 
 
 3.2.1.3 Label Bias 

 
 Label bias or annotation bias arises when labels used to train the model are not entirely correct. This will lead to the training labels not representing the true labels. Training labels can systematically deviate as a result of vagueness and cultural or individual differences ( Fahse et al., 2021 ) .
The authors in ( Sap et al., 2019 ) investigate annotation bias in African American English (AAE) tweets. When annotators were primed to consider dialect, their assessment of an AAE tweet being "offensive" or "not offensive" was shown to be biased.

 
 
 
 3.2.1.4 Co-variate Shift 

 
 This kind of bias arises when the distribution of the features used to train the model is different in the training and testing phase.
Imagine a model used to invite job interviewees that is trained using skills demanded by the industry a few years ago. If this model was to be used now, it would not be able to make accurate predictions since the skills demanded by the industry have shifted significantly
 ( Gu and Oelke, 2019 ) .

 
 
 
 3.2.1.5 Sampling Bias 

 
 Sampling bias can occur when sampling of subgroups is not random ( Mehrabi et al., 2021 ) . An example of this kind of bias can arise if a researcher who’s interested in learning the movie preferences of teens post on a social media group. This would primarily capture the preferences of teens who are members of that group. Sampling bias manifests here because teens who use social media and are part of the group might have different characteristics than teens who don’t even have social media accounts. To conclude, the researcher might not get an accurate representation of the movie preferences of teens as a whole.

 
 
 
 3.2.1.6 Specification Bias 

 
 This sort of bias arises during the specification of what comprises the input and output during a learning task ( Hellström et al., 2020 ) . The specifications are often prepared by system designers. If the design choices are misaligned with end goals, they can exhibit specification bias ( Tal, 2023 ) .

 
 
 
 3.2.1.7 Aggregation Bias 

 
 This bias arises when the true underlying patterns are misinterpreted by how the data is combined. Let’s imagine there is a smartwatch for fitness that keeps track of the number of steps that you take constantly throughout the day. The numbers of steps are then aggregated into weekly averages and fed to train an ML model. Aggregating the data weekly can make the model miss vital information, like predicting if the person reached their daily step goals, which can only be obtained from the short-term patterns of the data.

 
 
 
 3.2.1.8 Linking Bias 

 
 Linking bias occurs when true behavior of users is misrepresented due to the attributes of networks derived from user interactions, connections or activity ( Olteanu et al., 2019 ) . Let’s consider a social networking site where you connect with other users. The platform has some users who are highly active and have lots of connections, and others who are socially active in the physical world but not as active and have fewer connections on the platform. Linking bias will arise if the highly active users appear more central in the network due to their numerous connections. This could lead to an inaccurate representation of the actual social dynamic.

 
 
 
 3.2.1.9 Inherited Bias 

 
 According to ( Hellström et al., 2020 ) inherited bias arises when biased output from a tool is used as input for other machine learning algorithms. In simple terms, the new algorithms can inherit the bias from tools’ output. Let’s assume the output of an automated loan approval tool is used as input to a new machine learning model. If the loan approval tool is biased regarding race, then this bias will be inherited by the new model.

 
 
 
 3.2.1.10 Longitudinal Data Fallacy 

 
 This sort of bias can arise when cross-sectional analysis is performed to study temporal data. Let’s consider a study done to analyze the sales performance of item A in all stores of a specific area over 2 months. The results showed that the sales performance of item A was poor in all stores. However, after examining the data over 36 months, it was found that the sales of the same item in the same stores improved over time. This shows that a fallacy can emerge when temporal dynamics are not considered; instead, cross-sectional analysis is used.

 
 
 
 

#### 3.2.2. Human Bias

 
 Human bias can be present in the machine learning pipeline in various stages. It can be present as prejudices in the training data. Bias can also be introduced by humans in the development and deployment stages.

 
 
 3.2.2.1 Historical Bias 

 
 Historical bias can occur if models perpetuate bias present in the historical record. These biases can arise from sources including social prejudices, preconceived notions and unequal treatment of different individuals or groups. The authors in ( Ghosh and Caliskan, 2023 ) found that ChatGPT associated the action of cooking breakfast to female entities when asked to translate a gender-neutral sentence from Bengali to English. This can be due to backdated cultural and societal expectations of women to take on domestic roles like cooking, cleaning, etc.

 
 
 
 3.2.2.2 Population Bias 

 
 This sort of bias can arise when there are biased variations in user characteristics or demographics between the target population and the population of users represented in a data set ( Olteanu et al., 2019 ) . Let’s consider there is a mobile application to rent cars in a country. This application is primarily used by people living in the cities and not at all used by people residing in rural areas. If the data collected from this application is used to study the general preferences of the whole country’s population, then it may exhibit bias towards people inhabiting the city.

 
 
 
 3.2.2.3 Self-selection Bias 

 
 This kind of bias arises when subjects have the full right to decide if they want to participate in a study or not. Let’s consider a shopping site, where some active users provide reviews of the products they purchase. These users are self-selecting to give reviews. There are some users who purchase items, but never post reviews. If a recommendation system is created using these self-selected user reviews, then it might miss the preferences of the users who didn’t share their reviews. Thus, the system will be biased toward active users.

 
 
 
 3.2.2.4 Behavioral Bias 

 
 Behavioral bias occurs when user behavior across platforms or contexts displays systematic distortions ( Olteanu et al., 2019 ) . Jiang et al. ( Jiang et al., 2016 ) discusses how users can find it difficult to transfer learning from one platform to another, especially if the platforms are not very similar. This implies the user needs to change their behavior according to the characteristics of each platform since they are not very similar.

 
 
 
 3.2.2.5 Temporal Shift 

 
 This kind of bias is described as the systematic distortions that arise across behaviors over time or among different user populations ( Olteanu et al., 2019 ) . The authors in ( Gurjar et al., 2022 ) presented findings on how users start posting more content after they hit a popularity shock. Here, a temporal shift could include any changes in the users’ engagement patterns that concur with their newfound popularity.

 
 
 
 3.2.2.6 Content Production Bias 

 
 This bias occurs when user generated content has behavioral bias expressed as lexical, semantic, syntactic and structural differences ( Olteanu et al., 2019 ) . The authors in ( Paris et al., 2012 ) show how different communities in social media exhibit significant differences in language use for the content posted.

 
 
 
 3.2.2.7 Deployment Bias 

 
 Deployment bias emerges when a model is used or interpreted in a way that is not deemed appropriate when deployed to be used in the real world ( Fahse et al., 2021 ) . This will occur when a machine learning model is built and evaluated, assuming to be fully autonomous when, in reality, it is used in a complex socio-technical environment that follows human decisions ( Fahse et al., 2021 ) . As outlined by ( Lee and Singh, 2021 ) , an insurance company that used a fraud detection model resolved to use a human investigator feedback loop. When a fraud prediction was made by the model, human investigators would review the cases to validate the prediction ( Lee and Singh, 2021 ) . Deployment bias arises here because of how human validation interacts with the outcomes of the model. Humans checking the outcomes can unconsciously inject their biases into the validation process.

 
 
 
 3.2.2.8 Feedback Bias 

 
 This kind of bias arises when the output of a model influences features or inputs that are used for retraining or refining the model ( Fahse et al., 2021 ) . An example of this bias can occur if a movie recommendation system keeps suggesting movies based on the user’s past choices, which in turn limits exposure to other diverse options. The bias arises because the model uses the user’s choices to retrain itself to predict movies.

 
 
 
 3.2.2.9 Popularity Bias 

 
 Popularity Bias arises when well-liked items get more exposure ( Mehrabi et al., 2021 ) . The authors in ( Sushma Channamsetty, 2017 ) , shows that recommender systems suggest items solely relying on popularity and not aligning with user preferences.

 
 
 
 

#### 3.2.3. Model Bias

 
 This section describes how model design choices can cause biased predictions.

 
 
 3.2.3.1 Algorithmic Bias 

 
 This bias is caused by the algorithm itself and not the input data ( Baeza-Yates, 2018 ) . It could arise from structure, design and/or decision-making aspects of the algorithm itself. Let’s consider there is a facial recognition model that was trained to be fair and unbiased. When evaluating this model, it was found that it consistently misidentified individuals from a certain ethnicity. This sort of bias was not present in the training data, so it is possible that the bias may stem from the internal mechanism of this model. The algorithm may process facial features from cultural backgrounds differently and unknowingly cause this issue.

 
 
 
 3.2.3.2 Evaluation Bias 

 
 Evaluation bias emerges when the data used to test the model is not even close to what it would encounter in the real world. According to ( Fahse et al., 2021 ) , if a wrong benchmark set is selected, then it can lead to neglecting potential bias. Let’s consider a smile detector model trained using a data set without proper representations of Asian individuals.
When testing the model, if the benchmark used is unbalanced similar to the training set, then the bias against Asians will be overlooked.

 
 
 
 
 
 

## 4. Practical cases of unfairness in real-world setting

 
 Unfairness can be present in ML models in many real-world scenarios. This can manifest across various platforms and domains. For the last few decades, the idea of using automated tools to aid decision-making processes in numerous societal contexts has been made popular. This increased use of these tools has raised the question of fairness in the predictions provided by them. Models like these are present in various sectors, the subsections below describe some of these sectors in more detail.

 
 

### 4.1. Criminal Justice System

 
 The use of automated tools is now a norm in the criminal justice system. They are applied to various aspects of the justice system like law enforcement, corrections and court cases ( Gloria GONZÁLEZ FUSTER, 2020 ; Gstrein et al., 2019 ; Manning et al., 2018 ) . The most commonly used automated models in criminal justice are for offense profiling and risk assessment ( Christine Bannan, 2020 ) . The most widely known case of biased prediction in criminal justice is COMPAS (Correctional Offender Management Profiling for Alternative Sanctions), which was used as a recidivism indicator. COMPAS was widely used to predict the likelihood of a person to re-offend within two years ( Brennan et al., 2009 ) . The authors in ( Mattu et al., 2016 ) discuss how, despite having similar overall accuracy, the COMPAS algorithm discriminated against Black defendants by predicting a higher risk while favoring White defendants by assigning them a much lower risk. Similar to this case, legal researchers found that HART (Harm Assessment Risk Tool), which was used by the police to aid in decision making, prioritized assigning the high-risk individuals a low-risk score over assigning the low-risk individuals a high-risk score ( Marion Oswald and Barnes, 2018 ) . This, in turn, raised concerns about negative impacts on society from certain individuals who were given low-risk scores (but turned out to be high risks). Chouldechova ( Chouldechova, 2016 ) describes how it is necessary to adjust error rates in predictors to achieve fairness when predicting recidivism scores across different groups. Racial bias in forensic databases can also impact automated tools’ contributions to the criminal justice system ( Risher, 2011 ) . Gstrein et al. ( Gstrein et al., 2019 ) states how using black-box models to predict recidivism scores, guilt or innocence of convicts in court, lacks conformity with legal regulations since it raises concerns about transparency, fairness and following legal principles in general.

 
 
 

### 4.2. Hiring Employees

 
 Machine learning models have been utilized by a growing number of organizations to aid in making employment decisions, but it’s not without any challenges. If not evaluated properly, a seemingly fair model will perpetuate bias against certain groups when hiring predictions are made, especially if it is trained on historical data ( Barocas and Selbst, 2016 ; Christine Bannan, 2020 ) . The authors in ( Hanna et al., 2020 ) found gender and racial biases in Facebook’s advertisement delivery system for employment and housing advertisements. They also discuss how the delivery system used for Facebook advertisements can alter the actual audience based on the advertisement’s content. The authors of the papers ( Sweeney, 2013 ; Datta et al., 2015 ) discuss their findings on how Google’s advertising system was biased against certain subgroups when employment advertisements were made on the platform. Various research shows that major employers discriminate against females and some ethnic groups ( Bendick and Nunes, 2011 ; Johnson et al., 2016 ) . The authors in ( Raghavan et al., 2020 ) tested 18 vendors which used automated hiring models and found only a mere seven addressed bias in their hiring model.

 
 
 

### 4.3. Finance

 
 The financial industry increasingly employs automated tools to aid decision-making processes. However, their use of such tools comes with challenges that include ensuring compliance with consumer laws and minimizing disparate impacts for disadvantaged communities and making fair predictions in general. Sigalos et al. ( Sigalos, 2023 ) presents several cases of bias exacerbated by models when lending loans for fraud detection and investment recommendation.
In 2009, the author in ( Lieber, 2009 ) initiated a debate about fair algorithmic models used in fin-tech. They mentioned how American Express potentially used historical data to predict the future behaviors of current users. Various companies that use historical data to train their models are vulnerable to getting a biased model. There has been a lot of research that shows historical bias against Hispanic, Black and some minority communities for creditworthiness and interest rates ( Butler et al., 2020 ; Fuster et al., 2022 ) . Some credit card companies use "financial profiling" to judge a candidate’s creditworthiness ( News, 2009 ) . In this case, the bias arose when the prediction was based not only on the candidate’s financial behavior but also their shopping habits from certain stores. As transparency in these models has become a legal obligation for financial institutions, they are vulnerable to disparate impact ( Finocchiaro et al., 2021 ; Bartlett et al., 2022 ) .

 
 
 

### 4.4. Healthcare

 
 Discrimination in AI models used in healthcare is an increasing concern since it has serious consequences. Bias in such models can lead to inequitable practices and inaccurate diagnoses. Straw et al. ( Straw and Wu, 2022 ) demonstrated the presence of gender bias in a model that is used for diagnosing liver disease. A similar case emerged when an X-ray prediction model was shown to exhibit bias, when it wrongly predicted patients from certain groups like women, Black, Hispanic and young individuals to not need medical attention ( Cho, 2021 ) . Numerous other cases of biased models were shown to discriminate against certain ethnic groups ( Benjamin, 2019 ; Yogarajan et al., 2022 ; Bowles et al., 2013 ; Obermeyer et al., 2019 ) . Presumably, biased algorithms have been developed for tasks like heart surgery, kidney transplants, rectal and breast cancer, which seldom affected access to services and resource allocation ( Obermeyer et al., 2019 ) . The lack of inclusive datasets used to train models utilized in healthcare can add to the systematic under-representation ( Celi et al., 2022 ; Benjamens et al., 2020 ) .

 
 
 

### 4.5. Education

 
 Educational institutions employ machine learning models to achieve various goals, including setting the curriculum for students, deciding what resources they should receive and other critical decisions ( Christine Bannan, 2020 ) . These models have the potential to manifest bias in various ways. The scores given by e-rater (an automated essay scoring system) had some discrepancies with human-assigned scores, especially for some Asian individuals ( Bridgeman et al., 2009 ; Bridgeman et al., 2012 ) .
During COVID, students were awarded grades using ML models since they couldn’t appear for tests physically. The models used to score the students were shown to discriminate based on certain sensitive attributes ( Adams and McIntyre, 2020 ; Lee, 2020 ; Denes, 2023 ) . The authors in ( Baker and Hawn, 2021 ) presented a thorough survey about biased models used in education; those interested can check its content.

 
 
 

### 4.6. Others

 
 Bias can creep into several systems that are used in our daily lives. A popular ride-hailing service, Uber, was found to have a customer rating system that was a channel to workplace discrimination against drivers from certain ethnic backgrounds ( Rosenblat et al., 2017 ) . ChatGPT displayed bias against women when asked to translate sentences in certain languages ( Ghosh and Caliskan, 2023 ) . Bias was also detected in facial recognition systems like Amazon Rekognition, IBM Watson, Microsoft Azure face API and Google cloud vision API ( Wen and Holweg, 2023 ) . Bias in advertisement delivery is also a potential issue that can affect how users are impacted by the use of AI models. According to the authors in ( O’Brien and Ortutay, 2021 ) , the Facebook advertising algorithm was shown to display gender bias since men were more likely to get ads for jobs that are in male-dominated fields, and women were more likely to get ads that are in female-dominated domains. Bias in AI systems used in gaming is also reported as the form of discrimination against atypical behavior (for example, an autistic player) ( Fahey, 2011 ) and training on data from hardcore-player (who doesn’t represent the whole gaming population) ( Melhart et al., 2023 ) . Airbnb’s facial recognition system wasn’t able to match an Australian’s (with South-Asian heritage) selfie with the government-issued ID and sparked potential racial bias ( Iqbal, 2023 ) . This highlights the issue of bias against certain populations who might be under-represented in the training data. In January this year, the Nine network used an image of a female Australian politician in a news broadcast, and it was discovered that the image was digitally altered ( N/A, 2024 ) . They stated it was an unintended consequence of the AI resizer used. In 2015, Google’s photo service was blamed for labeling a photo of a Black individual as a gorilla, and the solution Google came up with after two years was to remove the word "gorilla" from the labels used for pictures ( Simonite, 2018 ) . The cases mentioned here are noteworthy, but covering all the different sectors that fell prey to biased results from AI models is out of the scope of the paper.

 
 
 
 

## 5. Ways to mitigate bias and promote Fairness

 
 It is crucial to remove bias from ML models, especially to make them responsible and ethical. Models can contain historical bias and discriminate against certain groups or individuals. To align with the principles of social justice, we need the models to be fair. Addressing bias can make the system more inclusive since it considers a diverse environment. Also, a model behaving unfairly will make the user less likely to accept them. Users will be able to trust and use fair and transparent systems with more confidence. Additionally, responsible systems will ensure that all laws and regulations related to fair and non-discriminatory practices are adhered to.

 
 
 Mitigating bias in ML models can be a complex task that requires an amalgamation of technical, ethical, explainable and organizational strategies. It is vital that these issues are addressed at different stages of the ML pipeline.

 
 

### 5.1. General Stages to mitigate bias

 
 Recently, extensive research has been conducted to promote fairness in machine learning models. These bias mitigation strategies can generally be categorized into three types: pre-processing, in-processing and post-processing. Figure 5 shows these three stages in the ML pipeline.

 
 
 Figure 5. The three crucial stages in the development of an ML model 
 
 

#### 5.1.1. Pre-processing

 
 Pre-processing strategies help remove bias from training data. These approaches try to ensure unfair patterns are reduced or removed from training data before it is used to feed the model. The impartiality and quality of training data influence the model’s effectiveness in making fair predictions ( Farayola et al., 2023 ) .

 
 
 

#### 5.1.2. In-processing

 
 This approach mitigates unfairness in the ML models training process and aims for bias-free outcomes. This strategy focuses on properly tuning and developing the algorithms used ( Farayola et al., 2023 ) . According to the authors in ( Farayola et al., 2023 ) , choosing an algorithm that is less vulnerable to bias and a proper hyper-parameter tuning process is vital during this stage.

 
 
 

#### 5.1.3. Post-processing

 
 If a model’s actual outcome is unfair with respect to one or more sensitive attributes, this strategy can mitigate that by modifying the model’s predictions. This strategy can be employed only when the model is done training and has made its initial predictions.
 
 

 
 
 It is important to realize that each of these three strategies has its own limitations. A single strategy alone may not be enough to tackle bias. So, it is important to decide which mechanism is best by taking under consideration the nature of the dataset, type of bias, fairness metric used and chosen model characteristics. It is also important to realize that some of these techniques can end up reducing accuracy ( Kleinberg et al., 2016 ) , so trade-offs between accuracy and fairness should be evaluated. Figure 6 , shows a taxonomy of all the mitigation strategies identified in the current literature.

 
 {forest} 
 Figure 6. Proposed Taxonomy of the Mitigating Strategies 
 
 
 
 

### 5.2. Removing Bias from Data

 
 De-biasing the data used to train a model is the first crucial step towards designing a fair and ethical model. It not only promotes fairness but also helps increase the accuracy of the model since it considers a more diverse dataset and, in turn, produces a more generalized outcome. Having a dataset without bias will lead to a responsible and inclusive model that will be useful for all members of our society.

 
 

#### 5.2.1. Disparate Impact remover

 
 This process involves manipulating and transforming data to address fairness. Initially, the potential disparate impacts on the sensitive attributes of the dataset are examined ( Feldman et al., 2015 ) . Next, the dataset is repaired by transforming the initial dataset to fix the issues identified. During this stage, rank is preserved to ensure the relative position of data points is not altered. The main aim of the approach is to address disparate impacts, which is important in reducing unintended consequences.The authors in ( Feldman et al., 2015 ) does this by modifying the attributes in the dataset to ensure the distribution of protected and unprotected groups are closer. The authors in ( Xu et al., 2020 ) take a different path, and use a modified version of differentially private stochastic gradient descent (DPSGD) to remove disparate impact.

 
 
 

#### 5.2.2. Sampling

 
 Sampling techniques can help address data bias by considering data misclassification and class imbalance. According to ( Miron et al., 2021 ) , data sampling can help effectively analyze data without losing its universality. Feature sampling is the process of selecting only relevant features for the model to train on and can also contribute to mitigating bias ( Miron et al., 2021 ) . Kamiran et al. ( Kamiran and Calders, 2012 ) describes two approaches of sampling, which include:

 
 • 
 
 Uniform Sampling: promotes fair dataset by replacing objects according to their respective weights.

 

 • 
 
 Preferential Sampling: mitigates discrimination by resampling data too close to the decision boundary.

 

 
 Many papers, including ( Kamiran and Calders, 2012 ; Leonelli et al., 2021 ; Agarwal et al., 2019 ; Chouldechova, 2016 ) , incorporate some form of sampling to ensure that the dataset used does not discriminate against any sensitive attribute groups. An interesting approach is taken by the authors in ( Wang and Singh, 2023 ) , where they randomly remove training examples that are related to the over-represented demographic groups and add samples to the under-represented groups using re-sampling.

 
 
 

#### 5.2.3. Re-weighting

 
 Similar to resampling, re-weighting is a pre-processing technique that can be used to promote fairness in datasets. This process simply assigns different weights to different data objects based on their relevance. This method is widely used for models that have the opportunity of weighted samples like SVMs. Kamiran et al. ( Kamiran and Calders, 2012 ) describes the weighting process, which efficiently up-weights data that are disadvantaged and down-weights data that can cause discrimination. According to ( Wang and Singh, 2021 ) , bias caused by missing values can be addressed using re-weighting techniques. This method can improve fairness, although there is a slight decrease in accuracy ( Wang and Singh, 2021 ) . Several studies describe how re-weighting techniques can be applied to address issues like selection bias and disparate impact discrimination ( Feldman et al., 2015 ; Chakraborty et al., 2020 ; Favier et al., 2023 ; Bellamy et al., 2019 ) . The authors in ( Lahoti et al., 2020 ) , take a different approach by integrating adversarial learning and re-weighting to improve the fairness of a model with unobserved sensitive attributes in the training data. The adversarial reweighed learning process mentioned by ( Lahoti et al., 2020 ) includes the components below:

 
 • 
 
 The main classification model

 

 • 
 
 An adversary for finding regions with high error rates

 

 • 
 
 Weights which are assigned by examining the error rates. A higher weight is assigned to error-prone regions to encourage the model to focus on getting better at being fair.

 

 
 This process helps encourage the model to improve predictions for all sensitive groups without knowing them explicitly.

 
 
 
 

### 5.3. Adversarial Learning

 
 Adversarial learning can be employed to mitigate bias in ML models indirectly. There are several ways in which adversarial learning can be used to de-bias. In ( Zhang et al., 2018 ) , the authors describe how the process of gradient-based adversarial debasing can help mitigate bias in models. The process in ( Zhang et al., 2018 ) can be summarized as:

 
 • 
 
 A predictor is used to get outcomes given X X (which is the main task)

 

 • 
 
 An adversary tries to predict the sensitive variable S S , based on the prediction made by the predictor.

 

 
 The predictor is trained in a way that not only fulfills its primary task but also maintains fairness by actively counteracting the effect of the adversary ( Zhang et al., 2018 ) . The authors in ( Ball-Burack et al., 2021 ; Grari et al., 2020 ; Favier et al., 2023 ) employ this type of adversarial de-biasing to promote fairness in their models.
 
 A loss-based adversarial de-biasing method will involve the process of optimizing the loss functions as well as the main task (main classification task) and auxiliary task (adversary for predicting sensitive attribute). The authors in ( Wang et al., 2019 ) implements two competing loss functions:

 
 • 
 
 Classifier loss that motivates the classifier to perform well.

 

 • 
 
 Critic loss which penalizes if it’s not able to predict the sensitive attribute from the classifier’s outcome.

 

 
 In ( Wu et al., 2021 ) , authors employ a regularization-based adversarial de-biasing method. Their model includes:

 
 • 
 
 A predictor (which is the main model).

 

 • 
 
 An attribute discriminator that predicts sensitive attributes based on model predictions.

 

 • 
 
 An orthogonality regularization term that motivates bias-free embedding without correlations.

 

 
 
 
 Yang et al. ( Yang et al., 2023b ) , describe how adversarial learning can be used to disentangle non-sensitive and sensitive features by optimizing Balanced Fairness Objective (BFO) for a recommendation model. Their process combines both loss functions and min-max optimization to achieve fairness in the model.

 
 
 

### 5.4. Causal Approaches

 
 For two random variables A A and B B , causality is the phenomenon when A A causes B B ( Loftus et al., 2018 ) . Making changes to A A will lead to a different outcome for the variable B B . In simple terms, causality describes the relationship between two events, where one event directly causes the other event. Understanding causal relationships between features is essential for tackling bias since sensitive factors that are out of an individual’s control should not be used to make decisions ( Loftus et al., 2018 ) . The process of using causal relationships to mitigate bias can be summarized as ( Zhang et al., 2023b ; Vig et al., 2020 ) :

 
 • 
 
 Identify and understand the causal pathways through which sensitive attributes can influence the model’s outcome. This will help figure out which levers can be adjusted.

 

 • 
 
 To disrupt the causal pathways, design interventions to reduce the bias caused by them. For this, techniques such as neuron adjustments, data augmentation, etc., can be used.

 

 • 
 
 To avoid any unintentional results from modifying the model, use only fine-grained adjustments, not anything drastic.

 

 
 Several studies have been conducted to obtain a fair model by using causal reasoning, including ( Nabi and Shpitser, 2018 ; Russell et al., 2017 ; Salimi et al., 2019 ; Kilbertus et al., 2018 ) . In ( Bareinboim and Pearl, 2016 ) , the authors merge three different causal analysis methods to resolve selection bias and confounding. Galhotra et al. ( 2017 ) introduces a novel causality-based metric to measure dissemination and uses it for software fairness testing. Counterfactual reasoning is an approach used to answer what would have happened if some factor was different (e.g. if the outcome would be the same for a person from a different race). Causal learning can be utilized to identify counterfactuals. The authors in ( Kusner et al., 2018 ) describe how a fair predictive model can be created by restricting the outcome to be impacted by any sensitive variable (which is identified using a causal graph) and using counterfactual reasoning to ensure hypothetical interventions are considered.

 
 
 

### 5.5. Regularization

 
 Regularization helps make the ML model more general. It adds constraints and penalties in the loss function to discourage unfair outcomes. The main objective of this technique is to prevent overfitting and create simpler models. The authors in ( Wang et al., 2021 ) constructs and optimizes a custom loss function by using a novel regularization term to introduce causality during the training process. Regularization has also been utilized to satisfy multiple definitions of fairness ( Kang et al., 2021 ; Tavakol, 2020 ) . Kamishima et al. ( Kamishima et al., 2012 ) introduced a prejudice remover, which penalizes the model if there are high correlations between the target and sensitive variables. Utilizing regularization for promoting fairness has significant potential and has been used in a de-biasing technique ( Sundararaman and Subramanian, 2022 ) , fairness-aware ranking system ( Memarrast et al., 2021 ) , mitigating bias in reinforcement-learning agents ( Yu et al., 2022 ) and enforcing fairness in deep learning models ( Olfat and Mintz, 2020 ) . Regularization is traditionally considered an in-processing technique, but can also be used in post-processing settings. Peterson et al. ( Petersen et al., 2021 ) utilized regularization to adjust the outputs of an existing model to ensure predictions for similar individuals are consistent.

 
 
 

### 5.6. Disregarding sensitive attribute

 
 It is important to ensure that the model doesn’t learn to associate sensitive attributes like gender, race, age etc. when producing results. Some existing literature has taken the approach of removing the sensitive variables completely. This has been achieved by methods like adversarial learning ( Poulain et al., 2023 ) . Various research has found ways to reduce or eliminate correlation between sensitive attributes and other attributes to mitigate any discrimination caused by them ( Calders and Verwer, 2010 ; Gitiaux and Rangwala, 2021 ; Du et al., 2021 ; Woodworth et al., 2017 ; Creager et al., 2019 ) .
Although masking sensitive attributes seems like a viable solution to stop the model from forming association with them, there might be correlated features in the data that indirectly contain information about the sensitive attributes ( Chakraborty et al., 2020 ) . So the technique of masking can prove to be ineffective against bias mitigation ( Kleinberg et al., 2018 ) .

 
 
 

### 5.7. Calibration

 
 Calibration can be described as the process of ensuring the proportions of positive predictions and proportions of positive examples are equal ( Dawid, 1982 ) . These techniques try to adjust the predictions of models to better reflect the true outcomes. There are two types of calibration techniques ( Xu et al., 2023 ) :

 
 • 
 
 Platt Scaling : Model’s raw scores are adjusted to represent probabilities more reliably.

 

 • 
 
 Isotonic Regression : Regardless of any bias in the model, if it predicts something to be likely then it should actually happen more often.

 

 
 The overall equality of positive predictions is desirable, but a well-calibrated model should maintain equal proportions of probability within different subgroups (eg. age, gender etc.). Fairness promoting calibration has been integrated by various researchers for tasks that include accurately reflecting scores for loan applications and risk assessments
 ( Crowson et al., 2016 ; Chouldechova, 2016 ; Liu et al., 2019 ; Noriega-Campero et al., 2019 ) and preventing discriminated decisions in MAB (multi-armed bandit) setting ( Liu et al., 2017 ) . The authors in ( Úrsula Hébert-Johnson et al., 2018 ) discuss an approach called multi-calibration, which goes beyond just calibrating probabilities for specific groups but targets certain subpopulations to mitigate bias and promote fairness and generalization.

 
 
 As mentioned in ( Kleinberg et al., 2016 ; Pleiss et al., 2017 ) , except for some exceptions (constrained cases) it is nearly impossible for a calibrated model also to satisfy equalized odds. Authors in ( Pleiss et al., 2017 ) suggest prioritizing either calibration or error rate (for fairness) instead of trying to achieve a balance of both.

 
 
 

### 5.8. Relabelling

 
 This technique aims to modify the ground truth values in the training dataset, to ensure fairness notions are satisfied ( Dunkelau, 2016 ) . Data massaging ( Kamiran and Calders, 2012 ; Kamiran and Calders, 2009 ) , is the process of taking a number of training data and changing their respective ground truths. According to ( Dunkelau, 2016 ) , data massaging allows any classifier to learn on a dataset that is fair and promotes group fairness. The process mentioned in ( Kamiran and Calders, 2009 ; Kamiran and Calders, 2012 ) for modifying labels to remove bias can be summarized as:

 
 • 
 
 A ranker to approximate the probability of the data points in the target class, without the sensitive attributes.

 

 • 
 
 Identifying the promotion candidates (candidates that are in -ve class but will be moved to +ve class) and demotion candidates (candidates that are in +ve class but will be moved to -ve class)

 

 • 
 
 Modify labels of an equal number of candidates by swapping them.

 

 • 
 
 Iteratively perform this modification process until a desired level of bias has been reduced.

 

 
 
 
 Numerous research uses re-labelling to promote fairness, including ( Tavakol, 2020 ; Kamiran and Calders, 2012 ; Yang et al., 2023b ) .
It is important to note that re-labelling does come with extra cost (finding distance for the data) ( Chakraborty et al., 2020 ) , so this is recommended when removing data points entirely will affect the model.

 
 
 

### 5.9. Hyper-parameter optimization

 
 This strategy can be used not only to increase a model’s performance but also to mitigate bias. This can be done by considering fairness criteria during the hyper-parameter tuning process. Dooley et al. ( Dooley et al., 2023 ) has conducted research to prove that architectures and hyper-parameters can have a notable impact on fairness in models. The authors in ( Yang et al., 2023a ) , use hyper-parameter-optimization (specifically grid search and five-fold cross validation), to find hyper-parameters values that not only increase the accuracy of the model, but also considers the fairness of the model. Finding the correct balance of accuracy and fairness is key when mitigating bias using hyper-parameter optimization, also known as fairness-aware hyper-parameter optimization (FHO). Cruz et al. ( F.Cruz et al., 2021 ) defines accuracy and fairness as a MOO (Multi Objective Optimization) problem. This objective can be satisfied by using the hyper-parameters tuning method to find a collection of hyper-parameters that optimize fairness by using a weighted-scalarization process ( F.Cruz et al., 2021 ) and Pareto methods ( Fetterman et al., 2023 ) .

 
 
 

### 5.10. Fairness through Reinforcement Learning

 
 Reinforcement Learning (RL) provides a unique way to mitigate bias in ML systems. This is a promising approach compared to supervised learning, especially in dynamic settings, due to its ability to learn from interacting with the environment. RL models can employ continuous learning because the model itself can adapt in real-time which can help mitigate bias by considering concept drifts and frequent data changes. Li et al. ( Li et al., 2020 ) utilized RL to explore and learn about demographic groups that are under-represented for a model to predict if a candidate is worth hiring or not. RL has also been used in improving public health strategies, including mitigating bias in precision contagion policies ( Atwood et al., 2019 ) and mitigating bias perpetuated from data for predicting if a patient is infected with COVID-19 ( Yang et al., 2023a ) . The authors in ( Wang and Deng, 2019 ) utilize RL to promote fairness of a facial recognition system by using the margin that is used in the loss functions for different racial groups. Both ( Zhang and Wang, 2021 ) and ( Singh et al., 2021 ) describe their shared focus on mitigating long-term biases from recommendation algorithms by using Reinforcement learning. RLHF (Reinforcement Learning from Human Feedback), where models adapt their behavior according to the feedback given by humans, has recently gained popularity in the quest of fair AI. The authors in ( Ouyang et al., 2022 ) and ( Elmalaki, 2021 ) use human feedback as a guide to developing the models, making them more efficient and adaptive.
The survey ( Gajane et al., 2022 ) , offers valuable insight on fair reinforcement learning.

 
 
 

### 5.11. Others

 
 This section describes processes used to mitigate bias that don’t fall under the umbrella of the previously mentioned sections but are worth mentioning. The authors in ( Shokrollahi, 2023 ) , describes an elaborate method to mitigate intersectional bias by employing techniques which include quantum computing alongside data augmentation, fairness-aware fine-tuning and RL. K-NN has also been used to mitigate bias for doing tasks like discovering potential discrimination in datasets ( Luong et al., 2011 ) and to identify local regions for a locally fair ensemble model ( Lässig et al., 2022 ) . The authors in ( Zhang et al., 2023a ) propose a novel approach to promote fairness in survival prediction models, by using censorship and ranking consistency. Qraitem et al. ( Qraitem et al., 2023 ) , introduce a new method called "bias mimicking" that trains a model using multiple sub-samples. Within each class in the dataset this process mimics bias distribution of other classes, which helps reduce correlations with sensitive attributes. Mahabadi et al. ( Mahabadi et al., 2020 ) proposes a novel approach to mitigating bias in NLP by having a secondary bias-aware model with the main NLP model. The bias-aware model’s predictions are used to adjust the loss function of the main model. Similar to this technique ( Jin et al., 2020 ) also uses multiple models to promote fairness. They introduce the UBM (Upstream Bias Mitigation) framework that uses transfer learning to address bias in models. An upstream model is bias-mitigated and then used to pre-train a downstream model.

 
 
 
 

## 6. How Users can be affected by unfair ML Systems

 
 Whenever AI models are created, the developers mostly think about how they can be used to achieve a certain purpose.
Usually the developers forget to think about what it will be like when using an AI model. They forget to cater for a good user experience. Bias in AI systems can profoundly impact users in several ways. It has various negative impacts like discrimination, limited opportunities, decline in reliance and privacy concerns. One of the most commonly used AI systems, ChatGPT, is susceptible to be a weapon of mass deception and can spread misinformation and create deep fakes ( Sison et al., 2023 ) . The use of ChatGPT or similar models as WMD (weapon of mass deception) can have many direct and indirect negative impacts on users, including erosion of trust and psychological manipulations. The authors in ( Lew and Schumacher, 2020 ) state that truly good user experiences (UX) are engaging, fun and addictive. The goal should be to develop systems that evoke good emotions and go beyond just user satisfaction ( Lew and Schumacher, 2020 ) .

 
 

### 6.1. Prejudiced Model against UX

 
 Models are susceptible to negativity bias because users tend to put more weight on any negative interactions with a model over neutral or positive experiences ( Experience, 2016 ) . The authors in ( Chen et al., 2023a ) conducted research on how bias in recommender systems can impact user experience. They point out that although these kinds of systems perform well when controlled tests are performed on them, they can lead to frustrated users by suggesting irrelevant and/or unfair recommendations. Users can stop trusting AI systems entirely if they consistently provide predictions that are biased and inappropriate ( Chen et al., 2023a ; Schwartz and Cohen, 2004 ) . Moreover, certain model’s biased predictions can lead to the user not being able to explore diverse options and limit their ability to find new items ( Chen et al., 2023a ) . The authors in ( Silberstein et al., 2020 ) discuss how models that are used for ad selection could lead to biased or irrelevant ads, which in turn triggers users to close them due to their negativity, ultimately leading to a not-so-pleasant experience. Popularity bias, mentioned in section 3.2.2.9 , can lead to user dissatisfaction since they only get to interact with "popular" posts ( Lacic et al., 2022 ) .
Targeting good UX requires mitigating bias in AI models. By ameliorating bias from models, we can hope to develop a user experience that is not only enjoyable but also fair.

 
 
 

### 6.2. Ethical Considerations

 
 The idea of AI systems either matching or surpassing human capabilities is a possibility that can only be controlled through the implementation of solid moral standards ( Gordon and Nyholm, 2021 ) . So, when developing and deploying such systems the designers need to carefully consider the numerous ethical concerns. Currently, there are no universally enforced laws outlining ethical considerations for AI models. But there are significant development in designing such guidelines and policies to serve as recommendations or best practices by developers and organizations. UNESCO has recommended a set of ethical guidelines, which follow core principles like fairness, transparency and sustainability ( N/A, 2016 ) . The European Union has their own regional regulations for AI applications and proposes ways to develop fair and transparent models ( Commission, 2019 ) . Several countries are working to create their own ethical guidelines for AI models, including Australia ( Resources, 2022 ) , Singapore ( on the Ethical Use of AI and Council), 2022 ) and Canada ( Secretariat, 2018 ) .
The authors in ( Jobin et al., 2019 ) , explore the existing ethical guidelines that are available for AI worldwide. They found some key principles that are considered globally. These key principles include

 
 

#### 6.2.1. Transparency to promote Explainability

 
 The goal of this concept is to make AI systems more understandable and interpretable ( Adadi and Berrada, 2018 ) . The authors in ( Ali et al., 2023 ) break down the concept of XAI (Explainable Artificial Intelligence) into 4 sub-concepts.

 
 • 
 
 Data Explainability: Understanding if there are biases that are perpetuating from the data used to train the model is essential. The designers should make the data collection and preparation process transparent so that the end users can understand it.

 

 • 
 
 Model Explainability: This is to ensure the internal details of the model are transparent and understandable to users.

 

 • 
 
 Post-hoc Explainability: This is used to explain an outcome that is already given by a complex model. This will help get an idea of the model’s reasoning but might not be able to reflect on the big picture.

 

 • 
 
 Explanation Assessment: This process aims to evaluate how well the XAI methods work in explaining predictions.

 

 
 This concept is used for various reasons including to minimize harm and improve AI modes
 ( N/A, 2017 ) , to comply with legal regulations ( Floridi et al., 2018 ) and to gain user trust ( AG, 2018 ) .

 
 
 

#### 6.2.2. Justice, Fairness and Equity

 
 Justice, fairness and equity are very complex concepts, especially in the realm of AI systems. The three concepts can be summarized as follows ( Jobin et al., 2019 ) :

 
 • 
 
 Justice: This concept is linked with following rules and putting a stop to bias and discrimination. It also includes ideas about inclusion, diversity and the right to appeal any unfair decisions.

 

 • 
 
 Fairness: This concept ensures outcomes of AI systems are unbiased and addresses its impact on society.

 

 • 
 
 Equity: This phenomenon ensures that an AI system is designed to predict outcomes that are equitable for every individual.

 

 
 
 
 

#### 6.2.3. Non-maleficence

 
 Non-maleficence upholds the principle of "do not harm" ( Floridi and Cowls, 2019 ) . Another intertwined concept is that of beneficence which is related to the principle of "do only good". Although these two principles sound quite similar, beneficence triggers an action, whereas non-maleficence may prompt not to ( Werthner et al., 2024 ) . For instance, if this concept was to be applied to health care,
beneficence would require a model to predict diagnosis for helping the patient and non-maleficence would require the model to avoid causing harm to the patient (like prescribing the wrong dosage of a medicine).

 
 
 

#### 6.2.4. Responsibility and Accountability

 
 Responsible AI ensures AI is developed, assessed and deployed in a safe and ethical way ( mesameki, 2024 ) . The authors in ( Dignum, 2019 ) extends this definition and states responsible AI doesn’t only ensure systems are developed in a good way but also for a good cause. They state there are three main actors that bear responsibility for the AI system’s actions: users, developers/designers, and authorities. At the end of the day, humans are still responsible for how AI systems behave. Accountability and responsibility are both closely related ideas in this context. Here accountability is the ability to justify decisions produced by the model. These concepts have two main aims, which include explanations and accountability for design.

 
 
 

#### 6.2.5. Privacy

 
 This is one of the most important concepts in ethical AI development. UNESCO lists privacy as one of the 10 core principles of AI Ethics ( N/A, 2016 ) . They state privacy should be protected and promoted throughout the life-cycle of an AI model. Jobin et al. ( Jobin et al., 2019 ) describe this concept as not only a fundamental value but also a right to be protected. It is often discussed in the context of data protection and security, which ensures users’ personal or sensitive information is safeguarded ( Jobin et al., 2019 ) . The authors in ( Werthner et al., 2024 ) , firmly believe that AI systems must consistently protect individual’s privacy and never store any data, especially those considered sensitive.
 
 
 Jobin et al. ( Jobin et al., 2019 ) explain how even though there is a general agreement on these principles there can be divergence in their respective interpretations. The way these principles are applied to different areas and implementation methods can also vary.
One thing to note is that although there are detailed ethical guidelines provided by private or public organizations, the main challenge lies in incorporating them into the development and deployment process of the system. This challenge can arise because of various reasons including vagueness of the guidelines, accuracy trade-offs, technical debt, feasibility and system complexity. Despite the challenges, designers should still aim to produce models that consider ethical guidelines.

 
 
 .

 
 
 
 
 

## 7. Challenges and Limitations

 
 Even though the approaches discussed earlier for mitigating bias show promising results, they have their own challenges and limitations. These approaches and techniques can prove to be unethical in settings where the accuracy of the model is crucial. Integrating fairness constraints to mitigate model bias can cause performance loss and vice versa ( Zliobaite, 2015 ; Hardt et al., 2016 ; Haas, 2019 ) . However, it is important to realize that how the model is evaluated is also a big concern. Model evaluation techniques can conceal or create new underlying biases. Establishing a balance between fairness and accuracy in such a model is a multi-faceted challenge. There is no one solution that is applicable to all cases. Thus, the ideal balance between these two concepts depends on their distinctive setting and application. Another challenge arises because there is no one universally accepted definition of fairness that can be utilized for all different AI models. The authors ( Barocas and Selbst, 2016 ) state that maintaining group fairness may result in unequal treatment towards individuals while emphasizing individual fairness may not handle systematic bias at the group level. It can also be challenging to recognize which definition of fairness is relevant for a particular context. Moreover, these definitions are updated and or changed over time and can pose a problem in the interpretation and fulfillment of each of them. Developers can overlook some intricate definitions because of their complexity. Additionally they tend to stick to satisfying demographic parity because of its simplicity.
 
 The approaches and techniques to mitigate bias mostly rely on statistical methods and might fail to encapsulate the nuances of the intricate details of human behaviors and decision-making. One of the ways to develop fair models is by incorporating inclusive datasets that represent the diversity of the world fairly. However, having a truly inclusive dataset is a challenge on its own. Some mitigating techniques employ human intervention. Although it is a promising solution, addressing bias this way has many limitations. These include the fact that human interventions can be subjective and can be prone to bias. Since humans are given the power to define bias and report it, this can be a conflict of interest. Also, it may lead to a lack of consistency since different individuals react in different ways, which can also be an issue. The authors in ( Calegari et al., 2023 ) state that although human judgment can help mitigate bias, the process of rectifying bias from models is easier than addressing human prejudice. A pressing issue, pointed out by the authors in ( Lin et al., 2021 ) , is how simply treating individuals equally is not enough to satisfy the notion of fairness, and the concept of equity should also be considered. Only considering the three stages (pre, in and post-processing) when developing mitigation techniques can be limiting. Microsoft has formulated a more modern approach with nine stages that provide a more sturdy process to evaluate fairness ( Amershi et al., 2019 ) . Regardless of the challenges and limitations, the development and deployment of fair and unbiased automated systems is a continuous process. Future work can address these challenges and limitations whilst continuing on new innovations that grasp the subtleties of fairness and equity in AI.

 
 
 

## 8. Conclusion 

 
 In this survey our focus lies in presenting the main ideas and research conducted to make AI models more fair and ethical. Even with having this specific focus, the amount of pertinent studies is vast, and the aim of this paper is not to provide an overview of all these efforts but rather to address key takeaways from several recurring themes and areas that offer valuable insights for our readers. We have established the significance of this survey by mentioning how it differs from other related surveys. To promote a better understanding of the concept of fairness and bias in the context of AI, we discuss the numerous definitions of fairness and types of bias.
An intriguing observation is that the majority of the biases are data-driven which can be mitigated using techniques like sampling, causal reasoning and re-weighting. Moreover, some of these biases can arise in multiple stages of the ML pipeline which puts emphasis on the need of a multi-pronged approach when trying to mitigate them. A common theme that we picked up was how researchers like to combine different mitigation strategies to gain a more holistic approach when addressing bias in models. Out of all the different mitigation strategies discussed, adversarial learning and regularization stand out for their wide-spread adoption. Additionally we also cover less common approaches with potential, like RL (Reinforcement Learning) and calibration.
 
 Next we dissect real-world cases of models that manifested discriminatory practices within different sectors. Across these sectors, a persistent pattern of bias against individuals from certain races (especially Black people) is observed. This finding is a jarring reminder of the need to investigate the causes of bias in AI models. Additionally, we talk about the ramifications biased models have on users. Exploring this domain is crucial since it helps researchers develop models that are not only fair but also equitable. We also delve into some ethical guidelines, recognizing them as somewhat of a roadmap for fair and responsible AI development. These aid developers, policy makers and researchers to work on models that are fair, transparent, accountable and explainable. A challenge when following these guidelines arises because of the vagueness and subject to interpretation nature of some of these guidelines. More research can be done in addressing the ambiguity of these guidelines. Lastly, we mentioned the challenges and limitations of the existing literature to motivate prospective advancements in the field of fair AI models.
This survey emphasizes on the big picture by covering the why, how and what of bias and fairness in AI models. Future research can explore certain areas that fell outside of this paper’s scope, which include (but not limited to), the standardized methods for testing for bias in models, less prominent mitigation strategies and methods to make models interpretable and explainable.
To bring it all together, developing and deploying fair automated models is crucial, especially in mitigating biases that can affect life-altering decisions. Due to the widespread utilization of AI, which expands to all aspects of our lives, demands for fair and ethical automated choices are bound to happen. Thus, researchers should aim to create fair models and try to alleviate bias from them.

 
 
 Acknowledgements. 
This material is based upon work supported by the Air Force Office of Scientific Research under award number FA2386-23-1-4003.

 
 
 

## References

 
 
 Adadi and Berrada (2018) 
 
Amina Adadi and Mohammed Berrada. 2018.

 
 Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI).

 
 IEEE Access 6 (2018), 52138–52160.

 
 
 https://doi.org/10.1109/ACCESS.2018.2870052 

 

 
 Adams and McIntyre (2020) 
 
Richard Adams and Niamh McIntyre. 2020.

 
 England A-level downgrades hit pupils from disadvantaged areas hardest.

 
 The Guardian N/A, N/A (Aug. 2020), N/A.

 
 

 https://www.theguardian.com/education/2020/aug/13/england-a-level-downgrades-hit-pupils-from-disadvantaged-areas-hardest 

 

 
 AG (2018) 
 
Deutsche Telekom AG. 2018.

 
 Guidelines for Artificial Intelligence.

 
 
 
 
 https://www.telekom.com/en/company/digital-responsibility/details/artificial-intelligence-ai-guideline-524366 

 

 
 Agarwal et al . (2019) 
 
Alekh Agarwal, Miroslav Dudík, and Zhiwei Steven Wu. 2019.

 
 Fair Regression: Quantitative Definitions and Reduction-based Algorithms.

 
 
 
 arXiv:1905.12843 [cs.LG]

 

 
 Ali et al . (2023) 
 
Sajid Ali, Tamer Abuhmed, Shaker El-Sappagh, Khan Muhammad, Jose M. Alonso-Moral, Roberto Confalonieri, Riccardo Guidotti, Javier Del Ser, Natalia Díaz-Rodríguez, and Francisco Herrera. 2023.

 
 Explainable Artificial Intelligence (XAI): What we know and what is left to attain Trustworthy Artificial Intelligence.

 
 Information Fusion 99 (2023), 101805.

 
 

 https://doi.org/10.1016/j.inffus.2023.101805 

 

 
 Amershi et al . (2019) 
 
Saleema Amershi, Andrew Begel, Christian Bird, Robert DeLine, Harald Gall, Ece Kamar, Nachiappan Nagappan, Besmira Nushi, and Thomas Zimmermann. 2019.

 
 Software engineering for machine learning: A case study. In 2019 IEEE/ACM 41st International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP) . IEEE, IEEE Press, Montreal, QC, Canada, 291–300.

 
 
 

 
 Atwood et al . (2019) 
 
James Atwood, Hansa Srinivasan, Yoni Halpern, and D Sculley. 2019.

 
 Fair treatment allocations in social networks.

 
 
 
 arXiv:1911.05489 [cs.SI]

 

 
 Baeza-Yates (2018) 
 
Ricardo Baeza-Yates. 2018.

 
 Bias on the Web.

 
 Commun. ACM 61, 6 (may 2018), 54–61.

 
 

 https://doi.org/10.1145/3209581 

 

 
 Baker and Hawn (2021) 
 
Ryan S Baker and Aaron Hawn. 2021.

 
 Algorithmic bias in education.

 
 International Journal of Artificial Intelligence in Education N/A, N/A (2021), 1–41.

 
 
 

 
 Ball-Burack et al . (2021) 
 
Ari Ball-Burack, Michelle Seng Ah Lee, Jennifer Cobbe, and Jatinder Singh. 2021.

 
 Differential Tweetment: Mitigating Racial Dialect Bias in Harmful Tweet Detection. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (Virtual Event, Canada) (FAccT ’21) . Association for Computing Machinery, New York, NY, USA, 116–128.

 
 

 https://doi.org/10.1145/3442188.3445875 

 

 
 Bansal (2022) 
 
Rajas Bansal. 2022.

 
 A Survey on Bias and Fairness in Natural Language Processing.

 
 
 
 arXiv:2204.09591 [cs.CL]

 

 
 Barda et al . (2020) 
 
Noam Barda, Gal Yona, Guy N Rothblum, Philip Greenland, Morton Leibowitz, Ran Balicer, Eitan Bachmat, and Noa Dagan. 2020.

 
 Addressing bias in prediction models by improving subpopulation calibration.

 
 Journal of the American Medical Informatics Association 28, 3 (11 2020), 549–558.

 
 

 https://doi.org/10.1093/jamia/ocaa283 
arXiv:https://academic.oup.com/jamia/article-pdf/28/3/549/36428833/ocaa283.pdf

 

 
 Bareinboim and Pearl (2016) 
 
Elias Bareinboim and Judea Pearl. 2016.

 
 Causal inference and the data-fusion problem.

 
 Proceedings of the National Academy of Sciences 113 (07 2016), 7345–7352.

 
 
 https://doi.org/10.1073/pnas.1510507113 

 

 
 Barocas and Selbst (2016) 
 
Solon Barocas and Andrew D. Selbst. 2016.

 
 Big Data’s Disparate Impact.

 
 
 
 
 https://doi.org/10.2139/ssrn.2477899 

 

 
 Bartlett et al . (2022) 
 
Robert Bartlett, Adair Morse, Richard Stanton, and Nancy Wallace. 2022.

 
 Consumer-lending discrimination in the FinTech Era.

 
 Journal of Financial Economics 143, 1 (2022), 30–56.

 
 

 https://doi.org/10.1016/j.jfineco.2021.05.047 

 

 
 Bellamy et al . (2019) 
 
R. K. E. Bellamy, K. Dey, M. Hind, S. C. Hoffman, S. Houde, K. Kannan, P. Lohia, J. Martino, S. Mehta, A. Mojsilović, S. Nagar, K. Natesan Ramamurthy, J. Richards, D. Saha, P. Sattigeri, M. Singh, K. R. Varshney, and Y. Zhang. 2019.

 
 AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias.

 
 IBM Journal of Research and Development 63, 4/5 (2019), 4:1–4:15.

 
 
 https://doi.org/10.1147/JRD.2019.2942287 

 

 
 Bendick and Nunes (2011) 
 
Marc Bendick and Ana Nunes. 2011.

 
 Developing the Research Basis for Controlling Bias in Hiring.

 
 Journal of Social Issues 68 (01 2011), 238–262.

 
 
 https://doi.org/10.1111/j.1540-4560.2012.01747.x 

 

 
 Benjamens et al . (2020) 
 
Stan Benjamens, Pranavsingh Dhunnoo, and Bertalan Meskó. 2020.

 
 The state of artificial intelligence-based FDA-approved medical devices and algorithms: an online database.

 
 NPJ digital medicine 3, 1 (2020), 118.

 
 
 

 
 Benjamin (2019) 
 
Ruha Benjamin. 2019.

 
 Assessing risk, automating racism.

 
 Science 366, 6464 (2019), 421–422.

 
 
 https://doi.org/10.1126/science.aaz3873 
arXiv:https://www.science.org/doi/pdf/10.1126/science.aaz3873

 

 
 Berk et al . (2017a) 
 
Richard Berk, Hoda Heidari, Shahin Jabbari, Matthew Joseph, Michael J. Kearns, Jamie Morgenstern, Seth Neel, and Aaron Roth. 2017a.

 
 A Convex Framework for Fair Regression.

 
 CoRR abs/1706.02409, N/A (2017), N/A.

 
 arXiv:1706.02409

 http://arxiv.org/abs/1706.02409 

 

 
 Berk et al . (2017b) 
 
Richard Berk, Hoda Heidari, Shahin Jabbari, Michael Kearns, and Aaron Roth. 2017b.

 
 Fairness in Criminal Justice Risk Assessments: The State of the Art.

 
 
 
 arXiv:1703.09207 [stat.ML]

 

 
 Blodgett et al . (2020) 
 
S.L. Blodgett, S. Barocas, H. Daumé, and H. Wallach. 2020.

 
 Language (Technology) is power: A critical survey of bias” in NLP. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (Eds.). Association for Computational Linguistics, Online, 5454–5476.

 
 

 
 ISSN: 0736-587X.

 

 
 Bowles et al . (2013) 
 
Tawnya L Bowles, Chung-Yuan Hu, Nancy Y You, John M Skibber, and Miguel A Rodriguez-Bigas andGeorge J Chang. 2013.

 
 An individualized conditional survival calculator for patients with rectal cancer - PubMed.

 
 
 
 
 https://pubmed-ncbi-nlm-nih-gov.ezproxy-b.deakin.edu.au/23575393/ 

 

 
 Brennan et al . (2009) 
 
Tim Brennan, William Dieterich, and Beate Ehret. 2009.

 
 Evaluating the Predictive Validity of the Compas Risk and Needs Assessment System.

 
 Criminal Justice and Behavior 36, 1 (2009), 21–40.

 
 
 https://doi.org/10.1177/0093854808326545 
arXiv:https://doi.org/10.1177/0093854808326545

 

 
 Bridgeman et al . (2009) 
 
Brent Bridgeman, Catherine Trapani, and Yigal Attali. 2009.

 
 Considering fairness and validity in evaluating automated scoring.

 
 N/A N/A, N/A (2009), N/A.

 
 
 

 
 Bridgeman et al . (2012) 
 
Brent Bridgeman, Catherine Trapani, and Yigal Attali. 2012.

 
 Comparison of Human and Machine Scoring of Essays: Differences by Gender, Ethnicity, and Country.

 
 Applied Measurement in Education 25 (01 2012), 27–40.

 
 
 https://doi.org/10.1080/08957347.2012.635502 

 

 
 Butler et al . (2020) 
 
Alexander W Butler, Erik J Mayer, and James Weston. 2020.

 
 Racial discrimination in the auto loan market.

 
 Available at SSRN N/A, N/A (2020), N/A.

 
 
 

 
 Calders and Verwer (2010) 
 
Toon Calders and Sicco Verwer. 2010.

 
 Three naive Bayes approaches for discrimination-free classification.

 
 Data Min. Knowl. Discov. 21 (09 2010), 277–292.

 
 
 https://doi.org/10.1007/s10618-010-0190-x 

 

 
 Calegari et al . (2023) 
 
Roberta Calegari, Gabriel G. Castañé, Michela Milano, and Barry O’Sullivan. 2023.

 
 Assessing and Enforcing Fairness in the AI Lifecycle. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI-23 , Edith Elkind (Ed.). International Joint Conferences on Artificial Intelligence Organization, Vienna, Austria, 6554–6562.

 
 
 https://doi.org/10.24963/ijcai.2023/735 

 
 Survey Track.

 

 
 Caton and Haas (2023) 
 
Simon Caton and Christian Haas. 2023.

 
 Fairness in Machine Learning: A Survey.

 
 ACM Comput. Surv. N/A, N/A (aug 2023), N/A.

 
 

 https://doi.org/10.1145/3616865 

 
 Just Accepted.

 

 
 Celi et al . (2022) 
 
Leo Anthony Celi, Jacqueline Cellini, Marie-Laure Charpignon, Edward Christopher Dee, Franck Dernoncourt, Rene Eber, William Greig Mitchell, Lama Moukheiber, Julian Schirmer, Julia Situ, Joseph Paguio, Joel Park, Judy Gichoya Wawira, Seth Yao, and for MIT Critical Data. 2022.

 
 Sources of bias in artificial intelligence that perpetuate healthcare disparities—A global review.

 
 PLOS Digital Health 1, 3 (March 2022), e0000022.

 
 

 https://doi.org/10.1371/journal.pdig.0000022 

 
 Publisher: Public Library of Science.

 

 
 Chakraborty et al . (2020) 
 
Joymallya Chakraborty, Suvodeep Majumder, Zhe Yu, and Tim Menzies. 2020.

 
 Fairway: a way to build fair ML software. In Proceedings of the 28th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE ’20, N/A) . ACM, Sacramento, California, N/A.

 
 
 https://doi.org/10.1145/3368089.3409697 

 

 
 Chaudhary et al . (2023) 
 
Bhushan Chaudhary, Anubha Pandey, Deepak Bhatt, and Darshika Tiwari. 2023.

 
 Practical Bias Mitigation through Proxy Sensitive Attribute Label Generation.

 
 
 
 arXiv:2312.15994 [cs.LG]

 

 
 Chen et al . (2023a) 
 
Jiawei Chen, Hande Dong, Xiang Wang, Fuli Feng, Meng Wang, and Xiangnan He. 2023a.

 
 Bias and Debias in Recommender System: A Survey and Future Directions.

 
 ACM Trans. Inf. Syst. 41, 3, Article 67 (feb 2023), 39 pages.

 
 

 https://doi.org/10.1145/3564284 

 

 
 Chen et al . (2023b) 
 
Pu Chen, Linna Wu, and Lei Wang. 2023b.

 
 AI Fairness in Data Management and Analytics: A Review on Challenges, Methodologies and Applications.

 
 Applied Sciences 13, 18 (2023), N/A.

 
 

 https://doi.org/10.3390/app131810258 

 

 
 Cheng et al . (2021) 
 
Lu Cheng, Kush R. Varshney, and Huan Liu. 2021.

 
 Socially Responsible AI Algorithms: Issues, Purposes, and Challenges.

 
 
 
 arXiv:2101.02032 [cs.CY]

 

 
 Chiappa and Gillam (2018) 
 
Silvia Chiappa and Thomas P. S. Gillam. 2018.

 
 Path-Specific Counterfactual Fairness.

 
 
 
 arXiv:1802.08139 [stat.ML]

 

 
 Cho (2021) 
 
Mildred K. Cho. 2021.

 
 Rising to the challenge of bias in health care AI.

 
 Nature Medicine 27, 12 (Dec. 2021), 2079–2081.

 
 

 https://doi.org/10.1038/s41591-021-01577-2 

 
 Number: 12 Publisher: Nature Publishing Group.

 

 
 Chouldechova (2016) 
 
Alexandra Chouldechova. 2016.

 
 Fair prediction with disparate impact: A study of bias in recidivism prediction instruments.

 
 
 
 arXiv:1610.07524 [stat.AP]

 

 
 Christine Bannan (2020) 
 
Margerite Blase Christine Bannan. 2020.

 
 Automated Intrusion, Systemic Discrimination.

 
 
 
 
 http://newamerica.org/oti/reports/automated-intrusion-systemic-discrimination/ 

 

 
 Commission (2019) 
 
European Commission. 2019.

 
 Ethics guidelines for trustworthy AI | Shaping Europe’s digital future.

 
 
 
 
 https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai 

 

 
 Corbett-Davies et al . (2017) 
 
Sam Corbett-Davies, Emma Pierson, Avi Feller, Sharad Goel, and Aziz Huq. 2017.

 
 Algorithmic Decision Making and the Cost of Fairness. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (Halifax, NS, Canada) (KDD ’17) . Association for Computing Machinery, New York, NY, USA, 797–806.

 
 

 https://doi.org/10.1145/3097983.3098095 

 

 
 Correa et al . (2022) 
 
Ramon Correa, Mahtab Shaan, Hari Trivedi, Bhavik Patel, Leo Anthony G. Celi, Judy W. Gichoya, and Imon Banerjee. 2022.

 
 A Systematic Review of ‘Fair’ AI Model Development for Image Classification and Prediction.

 
 Journal of Medical and Biological Engineering 42, 6 (Dec. 2022), 816–827.

 
 

 https://doi.org/10.1007/s40846-022-00754-z 

 

 
 Creager et al . (2019) 
 
Elliot Creager, David Madras, Jörn-Henrik Jacobsen, Marissa Weis, Kevin Swersky, Toniann Pitassi, and Richard Zemel. 2019.

 
 Flexibly fair representation learning by disentanglement. In International conference on machine learning . PMLR, PMLR, N/A, 1436–1445.

 
 
 

 
 Crowson et al . (2016) 
 
Cynthia S Crowson, Elizabeth J Atkinson, and Terry M Therneau. 2016.

 
 Assessing calibration of prognostic risk scores.

 
 Statistical Methods in Medical Research 25, 4 (2016), 1692–1706.

 
 
 https://doi.org/10.1177/0962280213497434 
arXiv:https://doi.org/10.1177/0962280213497434

 
 PMID: 23907781.

 

 
 Datta et al . (2015) 
 
Amit Datta, Michael Carl Tschantz, and Anupam Datta. 2015.

 
 Automated Experiments on Ad Privacy Settings: A Tale of Opacity, Choice, and Discrimination.

 
 
 
 arXiv:1408.6491 [cs.CR]

 

 
 Dawid (1982) 
 
A. P. Dawid. 1982.

 
 The Well-Calibrated Bayesian.

 
 J. Amer. Statist. Assoc. 77, 379 (1982), 605–610.

 
 
 https://doi.org/10.1080/01621459.1982.10477856 
arXiv:https://www.tandfonline.com/doi/pdf/10.1080/01621459.1982.10477856

 

 
 Deldjoo et al . (2023) 
 
Yashar Deldjoo, Dietmar Jannach, Alejandro Bellogin, Alessandro Difonzo, and Dario Zanzonelli. 2023.

 
 Fairness in recommender systems: research landscape and future directions.

 
 User Modeling and User-Adapted Interaction 34, N/A (April 2023), 59–108.

 
 

 https://doi.org/10.1007/s11257-023-09364-z 

 

 
 Denes (2023) 
 
Gyorgy Denes. 2023.

 
 A case study of using AI for General Certificate of Secondary Education (GCSE) grade prediction in a selective independent school in England.

 
 Computers and Education: Artificial Intelligence 4 (Jan. 2023), 100129.

 
 

 https://doi.org/10.1016/j.caeai.2023.100129 

 

 
 Dignum (2019) 
 
Virginia Dignum. 2019.

 
 Responsible artificial intelligence: how to develop and use AI in a responsible way . Vol. 2156.

 
 Springer, N/A.

 
 
 

 
 Dooley et al . (2023) 
 
Samuel Dooley, Rhea Sanjay Sukthanker, John P Dickerson, Colin White, Frank Hutter, and Micah Goldblum. 2023.

 
 Rethinking Bias Mitigation: Fairer Architectures Make for Fairer Face Recognition. In Thirty-seventh Conference on Neural Information Processing Systems . N/A, New Orleans, N/A.

 
 
 https://openreview.net/forum?id=1vzF4zWQ1E 

 

 
 Du et al . (2021) 
 
Mengnan Du, Subhabrata Mukherjee, Guanchu Wang, Ruixiang Tang, Ahmed Hassan Awadallah, and Xia Hu. 2021.

 
 Fairness via Representation Neutralization.

 
 
 
 arXiv:2106.12674 [cs.LG]

 

 
 Dunkelau (2016) 
 
Jannik Dunkelau. 2016.

 
 Fairness-Aware Machine Learning An Extensive Overview.

 
 N/A N/A, N/A (2016), N/A.

 
 
 https://api.semanticscholar.org/CorpusID:237483522 

 

 
 Duong and Conrad (2023) 
 
Manh Duong and Stefan Conrad. 2023.

 
 Towards Fairness and Privacy: A Novel Data Pre-processing Optimization Framework for Non-binary Protected Attributes .

 
 Springer Nature Singapore, Singapore, 105–120.

 
 

 https://doi.org/10.1007/978-981-99-8696-5_8 

 

 
 Elmalaki (2021) 
 
Salma Elmalaki. 2021.

 
 Fair-iot: Fairness-aware human-in-the-loop reinforcement learning for harnessing human variability in personalized iot. In Proceedings of the International Conference on Internet-of-Things Design and Implementation . Association for Computing Machinery, New York, NY, USA, 119–132.

 
 
 https://doi.org/10.1145/3450268.3453525 

 

 
 Experience (2016) 
 
World Leaders in Research-Based User Experience. 2016.

 
 The Negativity Bias in User Experience.

 
 
 
 
 https://www.nngroup.com/articles/negativity-bias-ux/ 

 

 
 Fahey (2011) 
 
Mike Fahey. 2011.

 
 Autistic Boy Branded A Cheater By Xbox Live [Update].

 
 
 
 
 https://kotaku.com/autistic-boy-branded-a-cheater-by-xbox-live-update-5743970 

 

 
 Fahse et al . (2021) 
 
Tobias Fahse, Viktoria Huber, and Benjamin van Giffen. 2021.

 
 Managing Bias in Machine Learning Projects.

 
 In N/A . N/A, N/A, 94–109.

 
 

 https://doi.org/10.1007/978-3-030-86797-3_7 

 

 
 Farayola et al . (2023) 
 
Michael Mayowa Farayola, Irina Tal, Bendechache Malika, Takfarinas Saber, and Regina Connolly. 2023.

 
 Fairness of AI in Predicting the Risk of Recidivism: Review and Phase Mapping of AI Fairness Techniques. In Proceedings of the 18th International Conference on Availability, Reliability and Security ( conf-loc , city Benevento /city , country Italy /country , /conf-loc ) (ARES ’23) . Association for Computing Machinery, New York, NY, USA, Article 76, 10 pages.

 
 

 https://doi.org/10.1145/3600160.3605033 

 

 
 Favier et al . (2023) 
 
Marco Favier, Toon Calders, Sam Pinxteren, and Jonathan Meyer. 2023.

 
 How to be fair? A study of label and selection bias.

 
 Machine Learning 112 (09 2023), 1–24.

 
 
 https://doi.org/10.1007/s10994-023-06401-1 

 

 
 F.Cruz et al . (2021) 
 
Andre F.Cruz, Pedro Saleiro, Catarina Belem, Carlos Soares, and Pedro Bizarro. 2021.

 
 Promoting Fairness through Hyperparameter Optimization. In 2021 IEEE International Conference on Data Mining (ICDM) . IEEE, Auckland, New Zealand, N/A.

 
 
 https://doi.org/10.1109/icdm51629.2021.00119 

 

 
 Feldman et al . (2015) 
 
Michael Feldman, Sorelle Friedler, John Moeller, Carlos Scheidegger, and Suresh Venkatasubramanian. 2015.

 
 Certifying and removing disparate impact.

 
 
 
 arXiv:1412.3756 [stat.ML]

 

 
 Ferrara (2023) 
 
Emilio Ferrara. 2023.

 
 Fairness And Bias in Artificial Intelligence: A Brief Survey of Sources, Impacts, And Mitigation Strategies.

 
 
 
 arXiv:2304.07683 [cs.CY]

 

 
 Fetterman et al . (2023) 
 
Abraham J. Fetterman, Ellie Kitanidis, Joshua Albrecht, Zachary Polizzi, Bryden Fogelman, Maksis Knutins, Bartosz Wróblewski, James B. Simon, and Kanjun Qiu. 2023.

 
 Tune As You Scale: Hyperparameter Optimization For Compute Efficient Training.

 
 
 
 arXiv:2306.08055 [cs.LG]

 

 
 Finocchiaro et al . (2021) 
 
Jessie Finocchiaro, Roland Maio, Faidra Monachou, Gourab K Patro, Manish Raghavan, Ana-Andreea Stoica, and Stratis Tsirtsis. 2021.

 
 Bridging Machine Learning and Mechanism Design towards Algorithmic Fairness. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (Virtual Event, Canada) (FAccT ’21) . Association for Computing Machinery, New York, NY, USA, 489–503.

 
 

 https://doi.org/10.1145/3442188.3445912 

 

 
 Fletcher et al . (2021) 
 
Richard Ribón Fletcher, Audace Nakeshimana, and Olusubomi Olubeko. 2021.

 
 Addressing Fairness, Bias, and Appropriate Use of Artificial Intelligence and Machine Learning in Global Health.

 
 Frontiers in Artificial Intelligence 3 (2021), N/A.

 
 

 https://www.frontiersin.org/articles/10.3389/frai.2020.561802 

 

 
 Floridi and Cowls (2019) 
 
Luciano Floridi and Josh Cowls. 2019.

 
 A Unified Framework of Five Principles for AI in Society.

 
 Harvard Data Science Review 1, 1 (jul 1 2019), N/A.

 
 
 
 https://hdsr.mitpress.mit.edu/pub/l0jsh9d1.

 

 
 Floridi et al . (2018) 
 
Luciano Floridi, Josh Cowls, Monica Beltrametti, Raja Chatila, Patrice Chazerand, Virginia Dignum, Christoph Lütge, Robert Madelin, Ugo Pagallo, Francesca Rossi, Burkhard Schafer, Peggy Valcke, and Effy Vayena. 2018.

 
 AI4People—An Ethical Framework for a Good AI Society: Opportunities, Risks, Principles, and Recommendations.

 
 Minds and Machines 28 (12 2018).

 
 
 https://doi.org/10.1007/s11023-018-9482-5 

 

 
 Foulds and Pan (2018) 
 
James R. Foulds and Shimei Pan. 2018.

 
 An Intersectional Definition of Fairness.

 
 CoRR abs/1807.08362 (2018), 1918–1921.

 
 arXiv:1807.08362

 http://arxiv.org/abs/1807.08362 

 

 
 Fowler (2023) 
 
Geoffrey A. Fowler. 2023.

 
 Perspective | Say what, Bard? What Google’s new AI gets right, wrong and weird.

 
 Washington Post N/A, N/A (April 2023), N/A.

 
 

 https://www.washingtonpost.com/technology/2023/03/21/google-bard/ 

 

 
 Fuster et al . (2022) 
 
Andreas Fuster, Paul Goldsmith-Pinkham, Tarun Ramadorai, and Ansgar Walther. 2022.

 
 Predictably unequal? The effects of machine learning on credit markets.

 
 The Journal of Finance 77, 1 (2022), 5–47.

 
 
 

 
 Gajane et al . (2022) 
 
Pratik Gajane, Akrati Saxena, Maryam Tavakol, George Fletcher, and Mykola Pechenizkiy. 2022.

 
 Survey on Fair Reinforcement Learning: Theory and Practice.

 
 
 
 arXiv:2205.10032 [cs.LG]

 

 
 Galhotra et al . (2017) 
 
Sainyam Galhotra, Yuriy Brun, and Alexandra Meliou. 2017.

 
 Fairness testing: testing software for discrimination. In Proceedings of the 2017 11th Joint Meeting on Foundations of Software Engineering (ESEC/FSE’17) . ACM, Germany, N/A.

 
 
 https://doi.org/10.1145/3106237.3106277 

 

 
 Garrido-Muñoz  et al . (2021) 
 
Ismael Garrido-Muñoz , Arturo Montejo-Ráez , Fernando Martínez-Santiago , and L. Alfonso Ureña-López . 2021.

 
 A Survey on Bias in Deep NLP.

 
 Applied Sciences 11, 7 (2021), N/A.

 
 

 https://doi.org/10.3390/app11073184 

 

 
 Ghosh and Caliskan (2023) 
 
S. Ghosh and A. Caliskan. 2023.

 
 ChatGPT Perpetuates Gender Bias in Machine Translation and Ignores Non-Gendered Pronouns: Findings across Bengali and Five other Low-Resource Languages. In "" . Association for Computing Machinery, New York, NY, USA, 901–912.

 
 

 https://doi.org/10.1145/3600211.3604672 

 

 
 Gitiaux and Rangwala (2021) 
 
Xavier Gitiaux and Huzefa Rangwala. 2021.

 
 Fair Representations by Compression.

 
 
 
 arXiv:2105.14044 [cs.LG]

 

 
 Gloria GONZÁLEZ FUSTER (2020) 
 
Vrije Universiteit Brussel Gloria GONZÁLEZ FUSTER. 2020.

 
 Artificial Intelligence and Law Enforcement - Impact on Fundamental Rights | Think Tank | European Parliament.

 
 
 
 
 https://www.europarl.europa.eu/thinktank/en/document/IPOL_STU(2020)656295 

 

 
 Gohar and Cheng (2023) 
 
Usman Gohar and Lu Cheng. 2023.

 
 A Survey on Intersectional Fairness in Machine Learning: Notions, Mitigation, and Challenges. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence ( conf-loc , city Macao /city , country P.R.China /country , /conf-loc ) (IJCAI ’23) . N/A, Macao, Article 742, 9 pages.

 
 

 https://doi.org/10.24963/ijcai.2023/742 

 

 
 Gordon and Nyholm (2021) 
 
John-Stewart Gordon and Sven Nyholm. 2021.

 
 Ethics of Artificial Intelligence.

 
 N/A N/A, N/A (02 2021).

 
 
 

 
 Grari et al . (2020) 
 
Vincent Grari, Sylvain Lamprier, and Marcin Detyniecki. 2020.

 
 Fairness-Aware Neural Rényi Minimization for Continuous Features. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20 , Christian Bessiere (Ed.). International Joint Conferences on Artificial Intelligence Organization, Yokohama, Japan, 2234–2240.

 
 
 https://doi.org/10.24963/ijcai.2020/309 

 

 
 Gstrein et al . (2019) 
 
Oskar Josef Gstrein, Anno Bunnik, and Andrej Zwitter. 2019.

 
 Ethical, Legal and Social Challenges of Predictive Policing.

 
 Católica Law Review 3, 3 (Dec. 2019), 77–98.

 
 

 

 
 Gu and Oelke (2019) 
 
Jindong Gu and Daniela Oelke. 2019.

 
 Understanding Bias in Machine Learning.

 
 
 
 arXiv:1909.01866 [cs.LG]

 

 
 Gurjar et al . (2022) 
 
Omkar Gurjar, Tanmay Bansal, Hitkul Jangra, Hemank Lamba, and Ponnurangam Kumaraguru. 2022.

 
 Effect of Popularity Shocks on User Behaviour.

 
 Proceedings of the International AAAI Conference on Web and Social Media 16 (May 2022), 253–263.

 
 

 https://doi.org/10.1609/icwsm.v16i1.19289 

 

 
 Haas (2019) 
 
Christian Haas. 2019.

 
 The Price of Fairness - A Framework to Explore Trade-offs in Algorithmic Fairness. In N/A . N/A, Munich, Germany.

 
 
 

 
 Hanna et al . (2020) 
 
Alex Hanna, Emily Denton, Andrew Smart, and Jamila Smith-Loud. 2020.

 
 Towards a critical race methodology in algorithmic fairness. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (FAT* ’20) . ACM, Spain, N/A.

 
 
 https://doi.org/10.1145/3351095.3372826 

 

 
 Hardt et al . (2016) 
 
Moritz Hardt, Eric Price, and Nathan Srebro. 2016.

 
 Equality of Opportunity in Supervised Learning. In Proceedings of the 30th International Conference on Neural Information Processing Systems (Barcelona, Spain) (NIPS’16) . Curran Associates Inc., Red Hook, NY, USA, 3323–3331.

 
 

 

 
 Hellström et al . (2020) 
 
Thomas Hellström, Virginia Dignum, and Suna Bensch. 2020.

 
 Bias in Machine Learning What is it Good (and Bad) for?

 
 CoRR abs/2004.00686 (2020), N/A.

 
 arXiv:2004.00686

 https://arxiv.org/abs/2004.00686 

 

 
 Hu et al . (2021) 
 
Hui Hu, Mike Borowczak, and Zhengzhang Chen. 2021.

 
 Privacy-Preserving Fair Machine Learning Without Collecting Sensitive Demographic Data. In 2021 International Joint Conference on Neural Networks (IJCNN) . N/A, Shenzhen, China, 1–9.

 
 
 https://doi.org/10.1109/IJCNN52387.2021.9534017 

 

 
 Iosifidis et al . (2020) 
 
Vasileios Iosifidis, Besnik Fetahu, and Eirini Ntoutsi. 2020.

 
 FAE: A Fairness-Aware Ensemble Framework.

 
 
 
 arXiv:2002.00695 [cs.AI]

 

 
 Iqbal (2023) 
 
Soaliha Iqbal. 2023.

 
 Airbnb’s AI Couldn’t Recognise This Woman’s South Asian Features It’s Part Of A Bigger Issue.

 
 
 
 
 https://www.pedestrian.tv/tech-gaming/airbnb-ai-racial-bias/ 

 

 
 Jiang et al . (2016) 
 
Meng Jiang, Peng Cui, Nicholas Jing Yuan, Xing Xie, and Shiqiang Yang. 2016.

 
 Little Is Much: Bridging Cross-Platform Behaviors through Overlapped Crowds.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 30, 1 (Feb. 2016), N/A.

 
 

 https://doi.org/10.1609/aaai.v30i1.10001 

 
 Number: 1.

 

 
 Jin et al . (2020) 
 
Xisen Jin, Francesco Barbieri, Aida Mostafazadeh Davani, Brendan Kennedy, Leonardo Neves, and Xiang Ren. 2020.

 
 Efficiently mitigating classification bias via transfer learning.

 
 arXiv preprint arXiv:2010.12864 abs/2010.12864, N/A (2020), N/A.

 
 
 

 
 Jobin et al . (2019) 
 
Anna Jobin, Marcello Ienca, and Effy Vayena. 2019.

 
 The global landscape of AI ethics guidelines.

 
 Nature Machine Intelligence 1, 9 (Sept. 2019), 389–399.

 
 

 https://doi.org/10.1038/s42256-019-0088-2 

 

 
 Johnson et al . (2016) 
 
Stefanie Johnson, David Hekman, and Elsa Chan. 2016.

 
 If There’s Only One Woman in Your Candidate Pool, There’s Statistically No Chance She’ll Be Hired.

 
 Harvard business review N/A, N/A (04 2016).

 
 
 

 
 Kamiran and Calders (2009) 
 
Faisal Kamiran and Toon Calders. 2009.

 
 Classifying without discriminating. In 2009 2nd International Conference on Computer, Control and Communication . N/A, Pakistan, 1 – 6.

 
 
 https://doi.org/10.1109/IC4.2009.4909197 

 

 
 Kamiran and Calders (2012) 
 
Faisal Kamiran and Toon Calders. 2012.

 
 Data preprocessing techniques for classification without discrimination.

 
 Knowledge and Information Systems 33, 1 (Oct. 2012), 1–33.

 
 

 https://doi.org/10.1007/s10115-011-0463-8 

 

 
 Kamishima et al . (2012) 
 
Toshihiro Kamishima, Shotaro Akaho, Hideki Asoh, and Jun Sakuma. 2012.

 
 Fairness-Aware Classifier with Prejudice Remover Regularizer. In N/A . Springer Berlin Heidelberg, Berlin, Heidelberg, 35–50.

 
 

 https://doi.org/10.1007/978-3-642-33486-3_3 

 

 
 Kang et al . (2021) 
 
Jian Kang, Tiankai Xie, Xintao Wu, Ross Maciejewski, and Hanghang Tong. 2021.

 
 Multifair: Multi-group fairness in machine learning.

 
 arXiv preprint arXiv:2105.11069 N/A, N/A (2021), N/A.

 
 
 

 
 Kaur et al . (2021) 
 
Davinder Kaur, Suleyman Uslu, and Arjan Durresi. 2021.

 
 Requirements for Trustworthy Artificial Intelligence – A Review. In Advances in Networked-Based Information Systems , Leonard Barolli, Kin Fun Li, Tomoya Enokido, and Makoto Takizawa (Eds.). Springer International Publishing, Cham, 105–115.

 
 

 

 
 Kaur et al . (2022) 
 
Davinder Kaur, Suleyman Uslu, Kaley J. Rittichier, and Arjan Durresi. 2022.

 
 Trustworthy Artificial Intelligence: A Review.

 
 ACM Comput. Surv. 55, 2, Article 39 (jan 2022), 38 pages.

 
 

 https://doi.org/10.1145/3491209 

 

 
 Khedr and Shoukry (2022) 
 
Haitham Khedr and Yasser Shoukry. 2022.

 
 CertiFair: A Framework for Certified Global Fairness of Neural Networks.

 
 
 
 arXiv:2205.09927 [cs.LG]

 

 
 Kilbertus et al . (2018) 
 
Niki Kilbertus, Mateo Rojas-Carulla, Giambattista Parascandolo, Moritz Hardt, Dominik Janzing, and Bernhard Schölkopf. 2018.

 
 Avoiding Discrimination through Causal Reasoning.

 
 
 
 arXiv:1706.02744 [stat.ML]

 

 
 Kleinberg et al . (2018) 
 
Jon Kleinberg, Jens Ludwig, Sendhil Mullainathan, and Ashesh Rambachan. 2018.

 
 Algorithmic Fairness.

 
 AEA Papers and Proceedings 108 (May 2018), 22–27.

 
 
 https://doi.org/10.1257/pandp.20181018 

 

 
 Kleinberg et al . (2016) 
 
Jon Kleinberg, Sendhil Mullainathan, and Manish Raghavan. 2016.

 
 Inherent Trade-Offs in the Fair Determination of Risk Scores.

 
 
 
 arXiv:1609.05807 [cs.LG]

 

 
 Kordzadeh and Ghasemaghaei (2022) 
 
Nima Kordzadeh and Maryam Ghasemaghaei. 2022.

 
 Algorithmic bias: review, synthesis, and future research directions.

 
 European Journal of Information Systems 31, 3 (2022), 388–409.

 
 
 https://doi.org/10.1080/0960085X.2021.1927212 
arXiv:https://doi.org/10.1080/0960085X.2021.1927212

 

 
 Kusner et al . (2018) 
 
Matt J. Kusner, Joshua R. Loftus, Chris Russell, and Ricardo Silva. 2018.

 
 Counterfactual Fairness.

 
 
 
 
 https://doi.org/10.48550/arXiv.1703.06856 

 
 arXiv:1703.06856 [cs, stat].

 

 
 Lacic et al . (2022) 
 
Emanuel Lacic, Leon Fadljevic, Franz Weissenboeck, Stefanie Lindstaedt, and Dominik Kowald. 2022.

 
 What Drives Readership? An Online Study on User Interface Types and Popularity Bias Mitigation in News Article Recommendations. In European Conference on Information Retrieval . Springer, Springer, Norway, 172–179.

 
 
 

 
 Lahoti et al . (2020) 
 
Preethi Lahoti, Alex Beutel, Jilin Chen, Kang Lee, Flavien Prost, Nithum Thain, Xuezhi Wang, and Ed H. Chi. 2020.

 
 Fairness without Demographics through Adversarially Reweighted Learning.

 
 
 
 arXiv:2006.13114 [cs.LG]

 

 
 Lahoti et al . (2019) 
 
Preethi Lahoti, Krishna P. Gummadi, and Gerhard Weikum. 2019.

 
 Operationalizing individual fairness with pairwise fair representations.

 
 Proceedings of the VLDB Endowment 13, 4 (Dec. 2019), 506–518.

 
 

 https://doi.org/10.14778/3372716.3372723 

 

 
 Lässig et al . (2022) 
 
Nico Lässig, Sarah Oppold, and Melanie Herschel. 2022.

 
 Metrics and algorithms for locally fair and accurate classifications using ensembles.

 
 Datenbank-Spektrum 22, 1 (2022), 23–43.

 
 
 

 
 Lee (2020) 
 
Georgina Lee. 2020.

 
 FactCheck: did England exam system favour private schools?

 
 
 
 
 https://www.channel4.com/news/factcheck/factcheck-did-england-exam-system-favour-private-schools 

 

 
 Lee and Singh (2021) 
 
Michelle Seng Ah Lee and Jatinder Singh. 2021.

 
 Risk Identification Questionnaire for Detecting Unintended Bias in the Machine Learning Development Lifecycle. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society (Virtual Event, USA) (AIES ’21) . Association for Computing Machinery, New York, NY, USA, 704–714.

 
 

 https://doi.org/10.1145/3461702.3462572 

 

 
 Lee et al . (2023) 
 
Nayeon Lee, Yejin Bang, Holy Lovenia, Samuel Cahyawijaya, Wenliang Dai, and Pascale Fung. 2023.

 
 Survey of Social Bias in Vision-Language Models.

 
 
 
 
 https://doi.org/10.48550/arXiv.2309.14381 

 
 Publication Title: arXiv e-prints ADS Bibcode: 2023arXiv230914381L.

 

 
 Leonelli et al . (2021) 
 
Sabina Leonelli, Rebecca Lovell, Benedict W Wheeler, Lora Fleming, and Hywel Williams. 2021.

 
 From FAIR data to fair data use: Methodological data fairness in health-related social media research.

 
 Big Data Society 8, 1 (2021), 20539517211010310.

 
 
 https://doi.org/10.1177/20539517211010310 
arXiv:https://doi.org/10.1177/20539517211010310

 

 
 Lew and Schumacher (2020) 
 
Gavin Lew and Robert M. Schumacher. 2020.

 
 AI and UX: Why Artificial Intelligence Needs User Experience .

 
 Apress, Berkeley, CA.

 
 

 https://doi.org/10.1007/978-1-4842-5775-3 

 

 
 Li et al . (2020) 
 
Danielle Li, Lindsey Raymond, and Peter Bergman. 2020.

 
 Hiring as Exploration.

 
 
 
 
 https://papers.ssrn.com/abstract=3683612 

 

 
 Lieber (2009) 
 
Ron Lieber. 2009.

 
 American Express Kept a (Very) Watchful Eye on Charges.

 
 The New York Times N/A, N/A (2009), N/A.

 
 

 https://www.nytimes.com/2009/01/31/your-money/credit-and-debit-cards/31money.html 

 

 
 Lin et al . (2021) 
 
Ying-Tung Lin, Tzu-Wei Hung, and Linus Ta-Lun Huang. 2021.

 
 Engineering Equity: How AI Can Help Reduce the Harm of Implicit Bias.

 
 Philosophy Technology 34, 1 (Nov. 2021), 65–90.

 
 

 https://doi.org/10.1007/s13347-020-00406-7 

 

 
 Liu et al . (2019) 
 
Lydia T. Liu, Max Simchowitz, and Moritz Hardt. 2019.

 
 The implicit fairness criterion of unconstrained learning.

 
 
 
 arXiv:1808.10013 [cs.LG]

 

 
 Liu et al . (2017) 
 
Yang Liu, Goran Radanovic, Christos Dimitrakakis, Debmalya Mandal, and David C. Parkes. 2017.

 
 Calibrated Fairness in Bandits.

 
 
 
 arXiv:1707.01875 [cs.LG]

 

 
 Loftus et al . (2018) 
 
Joshua R. Loftus, Chris Russell, Matt J. Kusner, and Ricardo Silva. 2018.

 
 Causal Reasoning for Algorithmic Fairness.

 
 
 
 arXiv:1805.05859 [cs.AI]

 

 
 Luong et al . (2011) 
 
Binh Thanh Luong, Salvatore Ruggieri, and Franco Turini. 2011.

 
 k-NN as an implementation of situation testing for discrimination discovery and prevention. In Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (San Diego, California, USA) (KDD ’11) . Association for Computing Machinery, New York, NY, USA, 502–510.

 
 

 https://doi.org/10.1145/2020408.2020488 

 

 
 Ma et al . (2023) 
 
Jing Ma, Ruocheng Guo, Aidong Zhang, and Jundong Li. 2023.

 
 Learning for Counterfactual Fairness from Observational Data. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23) . ACM, USA, N/A.

 
 
 https://doi.org/10.1145/3580305.3599408 

 

 
 Mahabadi et al . (2020) 
 
Rabeeh Karimi Mahabadi, Yonatan Belinkov, and James Henderson. 2020.

 
 End-to-End Bias Mitigation by Modelling Biases in Corpora.

 
 
 
 arXiv:1909.06321 [cs.CL]

 

 
 Manning et al . (2018) 
 
Matthew Manning, Gabriel T. W. Wong, Timothy Graham, Thilina Ranbaduge, Peter Christen, Kerry Taylor, Richard Wortley, Toni Makkai, and Pierre Skorich. 2018.

 
 Towards a ‘smart’ cost–benefit tool: using machine learning to predict the costs of criminal justice policy interventions.

 
 Crime Science 7, 1 (Oct. 2018), 12.

 
 

 https://doi.org/10.1186/s40163-018-0086-4 

 

 
 Marion Oswald and Barnes (2018) 
 
Sheena Urwin Marion Oswald, Jamie Grace and Geoffrey C. Barnes. 2018.

 
 Algorithmic risk assessment policing models: lessons from the Durham HART model and ‘Experimental’ proportionality.

 
 Information Communications Technology Law 27, 2 (2018), 223–250.

 
 
 https://doi.org/10.1080/13600834.2018.1458455 
arXiv:https://doi.org/10.1080/13600834.2018.1458455

 

 
 Mattu et al . (2016) 
 
Lauren Kirchner Mattu, Jeff Larson, Surya, and Julia Angwin. 2016.

 
 Machine Bias.

 
 
 
 
 https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing 

 

 
 Mehrabi et al . (2021) 
 
Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. 2021.

 
 A Survey on Bias and Fairness in Machine Learning.

 
 Comput. Surveys 54, 6 (July 2021), 115:1–115:35.

 
 

 https://doi.org/10.1145/3457607 

 

 
 Melhart et al . (2023) 
 
David Melhart, Julian Togelius, Benedikte Mikkelsen, Christoffer Holmgård, and Georgios N. Yannakakis. 2023.

 
 The Ethics of AI in Games.

 
 
 
 arXiv:2305.07392 [cs.HC]

 

 
 Memarrast et al . (2021) 
 
Omid Memarrast, Ashkan Rezaei, Rizal Fathony, and Brian Ziebart. 2021.

 
 Fairness for Robust Learning to Rank.

 
 
 
 arXiv:2112.06288 [cs.LG]

 

 
 mesameki (2024) 
 
mesameki. 2024.

 
 What is Responsible AI - Azure Machine Learning.

 
 
 
 
 https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai?view=azureml-api-2 

 

 
 Miron et al . (2021) 
 
Marius Miron, Songül Tolan, Emilia Gómez, and Carlos Castillo. 2021.

 
 Evaluating causes of algorithmic bias in juvenile criminal recidivism.

 
 Artif. Intell. Law 29, 2 (jun 2021), 111–147.

 
 

 https://doi.org/10.1007/s10506-020-09268-y 

 

 
 Mishler et al . (2021) 
 
Alan Mishler, Edward Kennedy, and Alexandra Chouldechova. 2021.

 
 Fairness in Risk Assessment Instruments: Post-Processing to Achieve Counterfactual Equalized Odds. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency . Association for Computing Machinery, New York, NY, USA, 386–400.

 
 
 https://doi.org/10.1145/3442188.3445902 

 

 
 N/A (2016) 
 
N/A. 2016.

 
 Ethics of Artificial Intelligence | UNESCO.

 
 
 
 
 https://www.unesco.org/en/artificial-intelligence/recommendation-ethics 

 

 
 N/A (2017) 
 
N/A. 2017.

 
 Artificial Intelligence Machine Learning: Policy Paper.

 
 
 
 
 https://www.internetsociety.org/resources/doc/2017/artificial-intelligence-and-machine-learning-policy-paper/ 

 

 
 N/A (2024) 
 
N/A. 2024.

 
 ’It should never happen again’: Victorian MP responds to Nine’s apology for digitally altered image.

 
 ABC News N/A, N/A (Jan. 2024), N/A.

 
 
 https://www.abc.net.au/news/2024-01-30/victorian-mp-georgie-purcell-altered-image/103403664 

 

 
 Nabi and Shpitser (2018) 
 
Razieh Nabi and Ilya Shpitser. 2018.

 
 Fair Inference on Outcomes.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 32, 1 (Apr. 2018), N/A.

 
 
 https://doi.org/10.1609/aaai.v32i1.11553 

 

 
 News (2009) 
 
A. B. C. News. 2009.

 
 ’GMA’ Gets Answers: Some Credit Card Companies Financially Profiling Customers.

 
 
 
 
 https://abcnews.go.com/GMA/TheLaw/gma-answers-credit-card-companies-financially-profiling-customers/story?id=6747461 

 

 
 Noriega-Campero et al . (2019) 
 
Alejandro Noriega-Campero, Michiel A. Bakker, Bernardo Garcia-Bulle, and Alex ’Sandy’ Pentland. 2019.

 
 Active Fairness in Algorithmic Decision Making. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society (Honolulu, HI, USA) (AIES ’19) . Association for Computing Machinery, New York, NY, USA, 77–83.

 
 

 https://doi.org/10.1145/3306618.3314277 

 

 
 Ntoutsi et al . (2020) 
 
Eirini Ntoutsi, Pavlos Fafalios, Ujwal Gadiraju, Vasileios Iosifidis, Wolfgang Nejdl, Maria-Esther Vidal, Salvatore Ruggieri, Franco Turini, Symeon Papadopoulos, Emmanouil Krasanakis, Ioannis Kompatsiaris, Katharina Kinder-Kurlanda, Claudia Wagner, Fariba Karimi, Miriam Fernandez, Harith Alani, Bettina Berendt, Tina Kruegel, Christian Heinze, Klaus Broelemann, Gjergji Kasneci, Thanassis Tiropanis, and Steffen Staab. 2020.

 
 Bias in data-driven artificial intelligence systems—An introductory survey.

 
 WIREs Data Mining and Knowledge Discovery 10, 3 (2020), e1356.

 
 

 https://doi.org/10.1002/widm.1356 

 
 _eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1002/widm.1356.

 

 
 Obermeyer et al . (2019) 
 
Ziad Obermeyer, Brian Powers, Christine Vogeli, and Sendhil Mullainathan. 2019.

 
 Dissecting racial bias in an algorithm used to manage the health of populations.

 
 Science (New York, N.Y.) 366, 6464 (Oct. 2019), 447–453.

 
 

 https://doi.org/10.1126/science.aax2342 

 

 
 Olfat and Mintz (2020) 
 
Matt Olfat and Yonatan Mintz. 2020.

 
 Flexible Regularization Approaches for Fairness in Deep Learning. In 2020 59th IEEE Conference on Decision and Control (CDC) . N/A, South Korea, 3389–3394.

 
 
 https://doi.org/10.1109/CDC42340.2020.9303736 

 

 
 Olteanu et al . (2019) 
 
Alexandra Olteanu, Carlos Castillo, Fernando Diaz, and Emre Kıcıman. 2019.

 
 Social data: Biases, methodological pitfalls, and ethical boundaries.

 
 Frontiers in big data 2 (2019), 13.

 
 
 

 
 on the Ethical Use of AI and Council) (2022) 
 
Advisory Council on the Ethical Use of AI and Data (Advisory Council). 2022.

 
 PDPC | Singapore’s Approach to AI Governance.

 
 
 
 
 https://www.pdpc.gov.sg/help-and-resources/2020/01/model-ai-governance-framework 

 

 
 Ouyang et al . (2022) 
 
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022.

 
 Training language models to follow instructions with human feedback.

 
 
 
 arXiv:2203.02155 [cs.CL]

 

 
 O’Brien and Ortutay (2021) 
 
Matt O’Brien and Barbara Ortutay. 2021.

 
 Study: Facebook delivers biased job ads, skewed by gender.

 
 
 
 
 https://apnews.com/article/discrimination-f62160cbbad4d72ce5250e6ef2222f5e 

 

 
 Paris et al . (2012) 
 
Cecile Paris, Paul Thomas, and Stephen Wan. 2012.

 
 Differences in Language and Style Between Two Social Media Communities.

 
 Proceedings of the International AAAI Conference on Web and Social Media 6, 1 (2012), 539–542.

 
 

 https://doi.org/10.1609/icwsm.v6i1.14307 

 
 Number: 1.

 

 
 Parraga et al . (2023) 
 
Otavio Parraga, Martin D. More, Christian M. Oliveira, Nathan S. Gavenski, Lucas S. Kupssinskü, Adilson Medronha, Luis V. Moura, Gabriel S. Simões, and Rodrigo C. Barros. 2023.

 
 Fairness in Deep Learning: A Survey on Vision and Language Research.

 
 ACM Comput. Surv. N/A, N/A (dec 2023), N/A.

 
 

 https://doi.org/10.1145/3637549 

 
 Just Accepted.

 

 
 Pavón Pérez (2022) 
 
Ángel Pavón Pérez. 2022.

 
 Bias in Artificial Intelligence Models in Financial Services. In Proceedings of the 2022 AAAI/ACM Conference on AI, Ethics, and Society (AIES ’22) . Association for Computing Machinery, New York, NY, USA, 908.

 
 

 https://doi.org/10.1145/3514094.3539561 

 

 
 Perrier (2021) 
 
Elija Perrier. 2021.

 
 Quantum Fair Machine Learning. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society (Virtual Event, USA) (AIES ’21) . Association for Computing Machinery, New York, NY, USA, 843–853.

 
 

 https://doi.org/10.1145/3461702.3462611 

 

 
 Pessach and Shmueli (2022) 
 
Dana Pessach and Erez Shmueli. 2022.

 
 A Review on Fairness in Machine Learning.

 
 ACM Comput. Surv. 55, 3, Article 51 (feb 2022), 44 pages.

 
 

 https://doi.org/10.1145/3494672 

 

 
 Pessach et al . (2024) 
 
Dana Pessach, Tamir Tassa, and Erez Shmueli. 2024.

 
 Fairness-Driven Private Collaborative Machine Learning.

 
 ACM Trans. Intell. Syst. Technol. 15, 2, Article 27 (feb 2024), 30 pages.

 
 

 https://doi.org/10.1145/3639368 

 

 
 Petersen et al . (2021) 
 
Felix Petersen, Debarghya Mukherjee, Yuekai Sun, and Mikhail Yurochkin. 2021.

 
 Post-processing for Individual Fairness.

 
 
 
 arXiv:2110.13796 [stat.ML]

 

 
 Piccininni (2022) 
 
Marco Piccininni. 2022.

 
 Counterfactual fairness: The case study of a food delivery platform’s reputational-ranking algorithm.

 
 Frontiers in Psychology 13 (2022), N/A.

 
 

 https://doi.org/10.3389/fpsyg.2022.1015100 

 

 
 Pleiss et al . (2017) 
 
Geoff Pleiss, Manish Raghavan, Felix Wu, Jon Kleinberg, and Kilian Q. Weinberger. 2017.

 
 On Fairness and Calibration.

 
 
 
 arXiv:1709.02012 [cs.LG]

 

 
 Poulain et al . (2023) 
 
R. Poulain, M.F. Bin Tarek, and R. Beheshti. 2023.

 
 Improving Fairness in AI Models on Electronic Health Records: The Case for Federated Learning Methods. In N/A . N/A, N/A, 1599–1608.

 
 

 https://doi.org/10.1145/3593013.3594102 

 

 
 Qraitem et al . (2023) 
 
Maan Qraitem, Kate Saenko, and Bryan A. Plummer. 2023.

 
 Bias Mimicking: A Simple Sampling Approach for Bias Mitigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) . IEEE Computer Society, Los Alamitos, CA, USA, 20311–20320.

 
 
 

 
 Raghavan et al . (2020) 
 
Manish Raghavan, Solon Barocas, Jon Kleinberg, and Karen Levy. 2020.

 
 Mitigating bias in algorithmic hiring: evaluating claims and practices. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (Barcelona, Spain) (FAT* ’20) . Association for Computing Machinery, New York, NY, USA, 469–481.

 
 

 https://doi.org/10.1145/3351095.3372828 

 

 
 Raman et al . (2021) 
 
Naveen Raman, Sanket Shah, and John Dickerson. 2021.

 
 Data-driven methods for balancing fairness and efficiency in ride-pooling.

 
 arXiv preprint arXiv:2110.03524 N/A (2021), N/A.

 
 
 

 
 Reimers et al . (2021) 
 
Christian Reimers, Paul Bodesheim, Jakob Runge, and Joachim Denzler. 2021.

 
 Towards Learning an Unbiased Classifier from Biased Data via Conditional Adversarial Debiasing.

 
 
 
 arXiv:2103.06179 [cs.CV]

 

 
 Resources (2022) 
 
Department of Industry Science and Resources. 2022.

 
 Australia’s AI Ethics Principles | Australia’s Artificial Intelligence Ethics Framework | Department of Industry Science and Resources.

 
 
 
 
 https://www.industry.gov.au/publications/australias-artificial-intelligence-ethics-framework/australias-ai-ethics-principles 

 

 
 Richardson and Gilbert (2021) 
 
Brianna Richardson and Juan E. Gilbert. 2021.

 
 A Framework for Fairness: A Systematic Review of Existing Fair AI Solutions.

 
 
 
 arXiv:2112.05700 [cs.AI]

 

 
 Risher (2011) 
 
Michael Risher. 2011.

 
 Racial Disparities in Databanking of DNA Profiles.

 
 N/A N/A, N/A (09 2011).

 
 

 https://doi.org/10.7312/columbia/9780231156974.003.0003 

 

 
 Rosenblat et al . (2017) 
 
Alex Rosenblat, Karen E.C. Levy, Solon Barocas, and Tim Hwang. 2017.

 
 Discriminating Tastes: Uber’s Customer Ratings as Vehicles for Workplace Discrimination.

 
 Policy Internet 9, 3 (2017), 256–279.

 
 

 https://doi.org/10.1002/poi3.153 

 
 _eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1002/poi3.153.

 

 
 Russell et al . (2017) 
 
Chris Russell, Matt J Kusner, Joshua Loftus, and Ricardo Silva. 2017.

 
 When Worlds Collide: Integrating Different Counterfactual Assumptions in Fairness. In Advances in Neural Information Processing Systems , I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Eds.), Vol. 30. Curran Associates, Inc., N?A.

 
 
 https://proceedings.neurips.cc/paper_files/paper/2017/file/1271a7029c9df08643b631b02cf9e116-Paper.pdf 

 

 
 Salimi et al . (2019) 
 
Babak Salimi, Luke Rodriguez, Bill Howe, and Dan Suciu. 2019.

 
 Interventional Fairness: Causal Database Repair for Algorithmic Fairness. In Proceedings of the 2019 International Conference on Management of Data (Amsterdam, Netherlands) (SIGMOD ’19) . Association for Computing Machinery, New York, NY, USA, 793–810.

 
 

 https://doi.org/10.1145/3299869.3319901 

 

 
 Salvador et al . (2021) 
 
Tiago Salvador, Stephanie Cairns, Vikram S. Voleti, Noah Marshall, and Adam M. Oberman. 2021.

 
 Bias Mitigation of Face Recognition Models Through Calibration.

 
 ArXiv abs/2106.03761 (2021).

 
 
 https://api.semanticscholar.org/CorpusID:235358539 

 

 
 Sap et al . (2019) 
 
Maarten Sap, Dallas Card, Saadia Gabriel, Yejin Choi, and Noah A. Smith. 2019.

 
 The Risk of Racial Bias in Hate Speech Detection. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics , Anna Korhonen, David Traum, and Lluís Màrquez (Eds.). Association for Computational Linguistics, Florence, Italy, 1668–1678.

 
 
 https://doi.org/10.18653/v1/P19-1163 

 

 
 Saxena et al . (2019) 
 
Nripsuta Ani Saxena, Karen Huang, Evan DeFilippis, Goran Radanovic, David C. Parkes, and Yang Liu. 2019.

 
 How Do Fairness Definitions Fare? Examining Public Attitudes Towards Algorithmic Definitions of Fairness. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society (Honolulu, HI, USA) (AIES ’19) . Association for Computing Machinery, New York, NY, USA, 99–106.

 
 

 https://doi.org/10.1145/3306618.3314248 

 

 
 Schwartz and Cohen (2004) 
 
Zvi Schwartz and Eli Cohen. 2004.

 
 Hotel Revenue-management Forecasting: Evidence of Expert-judgment Bias.

 
 Cornell Hotel and Restaurant Administration Quarterly 45, 1 (2004), 85–98.

 
 
 https://doi.org/10.1177/0010880403260110 
arXiv:https://doi.org/10.1177/0010880403260110

 

 
 Secretariat (2018) 
 
Treasury Board of Canada Secretariat. 2018.

 
 Responsible use of artificial intelligence (AI).

 
 
 
 
 https://www.canada.ca/en/government/system/digital-government/digital-government-innovations/responsible-use-ai.html 

 
 Last Modified: 2024-02-20.

 

 
 Shankar et al . (2017) 
 
Shreya Shankar, Yoni Halpern, Eric Breck, James Atwood, Jimbo Wilson, and D. Sculley. 2017.

 
 No Classification without Representation: Assessing Geodiversity Issues in Open Data Sets for the Developing World.

 
 
 
 arXiv:1711.08536 [stat.ML]

 

 
 Shokrollahi (2023) 
 
Omid Shokrollahi. 2023.

 
 Intersectional Bias Mitigation in Pre-trained Language Models: A Quantum-Inspired Approach. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management ( conf-loc , city Birmingham /city , country United Kingdom /country , /conf-loc ) (CIKM ’23) . Association for Computing Machinery, New York, NY, USA, 5181–5184.

 
 

 https://doi.org/10.1145/3583780.3616003 

 

 
 Siddique et al . (2024) 
 
Sunzida Siddique, Mohd Ariful Haque, Roy George, Kishor Datta Gupta, Debashis Gupta, and Md Jobair Hossain Faruk. 2024.

 
 Survey on Machine Learning Biases and Mitigation Techniques.

 
 Digital 4, 1 (2024), 1–68.

 
 

 https://doi.org/10.3390/digital4010001 

 

 
 Sigalos (2023) 
 
Ryan Browne Sigalos, MacKenzie. 2023.

 
 A.I. has a discrimination problem. In banking, the consequences can be severe.

 
 
 
 
 https://www.cnbc.com/2023/06/23/ai-has-a-discrimination-problem-in-banking-that-can-be-devastating.html 

 

 
 Silberstein et al . (2020) 
 
Natalia Silberstein, Oren Somekh, Yair Koren, Michal Aharon, Dror Porat, Avi Shahar, and Tingyi Wu. 2020.

 
 Ad Close Mitigation for Improved User Experience in Native Advertisements. In Proceedings of the 13th International Conference on Web Search and Data Mining (Houston, TX, USA) (WSDM ’20) . Association for Computing Machinery, New York, NY, USA, 546–554.

 
 

 https://doi.org/10.1145/3336191.3371798 

 

 
 Simonite (2018) 
 
Tom Simonite. 2018.

 
 When It Comes to Gorillas, Google Photos Remains Blind.

 
 Wired N/A, N/A (2018), N/A.

 
 

 https://www.wired.com/story/when-it-comes-to-gorillas-google-photos-remains-blind/ 

 
 Section: tags.

 

 
 Singh et al . (2021) 
 
Ashudeep Singh, Yoni Halpern, Nithum Thain, Konstantina Christakopoulou, Ed H. Chi, Jilin Chen, and Alex Beutel. 2021.

 
 Building Healthy Recommendation Sequences for Everyone: A Safe Reinforcement Learning Approach. In N/A . N/A, N/A, N/A.

 
 
 https://www.semanticscholar.org/paper/Building-Healthy-Recommendation-Sequences-for-A-Singh-Halpern/b1335578a35a3e74f6686533f2509cc96f25bcf7 

 

 
 Sison et al . (2023) 
 
Alejo Jose G. Sison, Marco Tulio Daza, Roberto Gozalo-Brizuela, and Eduardo C. Garrido-Merchán. 2023.

 
 ChatGPT: More than a Weapon of Mass Deception, Ethical challenges and responses from the Human-Centered Artificial Intelligence (HCAI) perspective.

 
 
 
 arXiv:2304.11215 [cs.CY]

 

 
 Straw and Wu (2022) 
 
Isabel Straw and Honghan Wu. 2022.

 
 Investigating for bias in healthcare algorithms: a sex-stratified analysis of supervised machine learning models in liver disease prediction.

 
 BMJ Health Care Informatics 29 (04 2022), 100457.

 
 
 https://doi.org/10.1136/bmjhci-2021-100457 

 

 
 Sundararaman and Subramanian (2022) 
 
Dhanasekar Sundararaman and Vivek Subramanian. 2022.

 
 Debiasing Gender Bias in Information Retrieval Models.

 
 
 
 arXiv:2208.01755 [cs.CL]

 

 
 Suresh and Guttag (2021) 
 
Harini Suresh and John Guttag. 2021.

 
 A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle. In Equity and Access in Algorithms, Mechanisms, and Optimization (EAAMO ’21) . ACM, N/A, N/A.

 
 
 https://doi.org/10.1145/3465416.3483305 

 

 
 Sushma Channamsetty (2017) 
 
Michael D. Ekstrand Sushma Channamsetty. 2017.

 
 Recommender Response to Diversity and Popularity Bias in User Profiles.

 
 
 
 
 https://aaai.org/papers/657-flairs-2017-15524/ 

 

 
 Sweeney (2013) 
 
Latanya Sweeney. 2013.

 
 Discrimination in Online Ad Delivery: Google ads, black names and white names, racial discrimination, and click advertising.

 
 Queue 11, 3 (mar 2013), 10–29.

 
 

 https://doi.org/10.1145/2460276.2460278 

 

 
 Tal (2023) 
 
Eran Tal. 2023.

 
 Target specification bias, counterfactual prediction, and algorithmic fairness in healthcare. In Proceedings of the 2023 AAAI/ACM Conference on AI, Ethics, and Society (AIES ’23) . Association for Computing Machinery, New York, NY, USA, 312–321.

 
 

 https://doi.org/10.1145/3600211.3604678 

 

 
 Tang et al . (2023) 
 
Zeyu Tang, Jiji Zhang, and Kun Zhang. 2023.

 
 What-is and How-to for Fairness in Machine Learning: A Survey, Reflection, and Perspective.

 
 ACM Comput. Surv. 55, 13s, Article 299 (jul 2023), 37 pages.

 
 

 https://doi.org/10.1145/3597199 

 

 
 Tavakol (2020) 
 
Maryam Tavakol. 2020.

 
 Fair classification with counterfactual learning. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval . N/A, China, 2073–2076.

 
 
 

 
 Timmaraju et al . (2023) 
 
Aditya Srinivas Timmaraju, Mehdi Mashayekhi, Mingliang Chen, Qi Zeng, Quintin Fettes, Wesley Cheung, Yihan Xiao, Manojkumar Rangasamy Kannadasan, Pushkar Tripathi, Sean Gahagan, Miranda Bogen, and Rob Roudani. 2023.

 
 Towards Fairness in Personalized Ads Using Impression Variance Aware Reinforcement Learning. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23) . ACM, USA, N/A.

 
 
 https://doi.org/10.1145/3580305.3599916 

 

 
 Timmons et al . (2023) 
 
Adela C. Timmons, Jacqueline B. Duong, Natalia Simo Fiallo, Theodore Lee, Huong Phuc Quynh Vo, Matthew W. Ahle, Jonathan S. Comer, LaPrincess C. Brewer, Stacy L. Frazier, and Theodora Chaspari. 2023.

 
 A Call to Action on Assessing and Mitigating Bias in Artificial Intelligence Applications for Mental Health.

 
 Perspectives on Psychological Science 18, 5 (2023), 1062–1096.

 
 
 https://doi.org/10.1177/17456916221134490 
arXiv:https://doi.org/10.1177/17456916221134490

 
 PMID: 36490369.

 

 
 Ueda et al . (2024) 
 
Daiju Ueda, Taichi Kakinuma, Shohei Fujita, Koji Kamagata, Yasutaka Fushimi, Rintaro Ito, Yusuke Matsui, Taiki Nozaki, Takeshi Nakaura, Noriyuki Fujima, Fuminari Tatsugami, Masahiro Yanagawa, Kenji Hirata, Akira Yamada, Takahiro Tsuboyama, Mariko Kawamura, Tomoyuki Fujioka, and Shinji Naganawa. 2024.

 
 Fairness of artificial intelligence in healthcare: review and recommendations.

 
 Japanese Journal of Radiology 42, 1 (Jan. 2024), 3–15.

 
 

 https://doi.org/10.1007/s11604-023-01474-3 

 

 
 van Breugel et al . (2021) 
 
Boris van Breugel, Trent Kyono, Jeroen Berrevoets, and Mihaela van der Schaar. 2021.

 
 DECAF: Generating Fair Synthetic Data Using Causally-Aware Generative Networks.

 
 
 
 arXiv:2110.12884 [cs.LG]

 

 
 Vasudevan and Kenthapadi (2020) 
 
Sriram Vasudevan and Krishnaram Kenthapadi. 2020.

 
 LiFT: A Scalable Framework for Measuring Fairness in ML Applications.

 
 Proceedings of the 29th ACM International Conference on Information Knowledge Management N/A, N/A (2020), N/A.

 
 
 https://api.semanticscholar.org/CorpusID:221139572 

 

 
 Verma and Rubin (2018) 
 
Sahil Verma and Julia Rubin. 2018.

 
 Fairness Definitions Explained. In Proceedings of the International Workshop on Software Fairness (Gothenburg, Sweden) (FairWare ’18) . Association for Computing Machinery, New York, NY, USA, 1–7.

 
 

 https://doi.org/10.1145/3194770.3194776 

 

 
 Vig et al . (2020) 
 
Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Simas Sakenis, Jason Huang, Yaron Singer, and Stuart Shieber. 2020.

 
 Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias.

 
 
 
 arXiv:2004.12265 [cs.CL]

 

 
 Wang and Deng (2019) 
 
Mei Wang and Weihong Deng. 2019.

 
 Mitigate Bias in Face Recognition using Skewness-Aware Reinforcement Learning.

 
 
 
 arXiv:1911.10692 [cs.CV]

 

 
 Wang et al . (2019) 
 
T. Wang, J. Zhao, M. Yatskar, K.-W. Chang, and V. Ordonez. 2019.

 
 Balanced datasets are not enough: Estimating and mitigating gender bias in deep image representations. In N/A , Vol. 2019-October. N/A, N/A, 5309–5318.

 
 

 https://doi.org/10.1109/ICCV.2019.00541 

 
 ISSN: 1550-5499.

 

 
 Wang and Singh (2021) 
 
Yanchen Wang and Lisa Singh. 2021.

 
 Analyzing the impact of missing values and selection bias on fairness.

 
 International Journal of Data Science and Analytics 12, 2 (Aug. 2021), 101–119.

 
 

 https://doi.org/10.1007/s41060-021-00259-z 

 

 
 Wang and Singh (2023) 
 
Yanchen Wang and Lisa Singh. 2023.

 
 Mitigating demographic bias of machine learning models on social media. In Proceedings of the 3rd ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization ( conf-loc , city Boston /city , state MA /state , country USA /country , /conf-loc ) (EAAMO ’23) . Association for Computing Machinery, New York, NY, USA, Article 24, 12 pages.

 
 

 https://doi.org/10.1145/3617694.3623244 

 

 
 Wang et al . (2021) 
 
Zhao Wang, Kai Shu, and Aron Culotta. 2021.

 
 Enhancing Model Robustness and Fairness with Causality: A Regularization Approach.

 
 
 
 arXiv:2110.00911 [cs.LG]

 

 
 Wen and Holweg (2023) 
 
Yuni Wen and Matthias Holweg. 2023.

 
 A phenomenological perspective on AI ethical failures: The case of facial recognition technology.

 
 AI SOCIETY N/A, N/A (April 2023), N/A.

 
 

 https://doi.org/10.1007/s00146-023-01648-7 

 

 
 Werthner et al . (2024) 
 
Hannes Werthner, Carlo Ghezzi, Jeff Kramer, Julian Nida-Rümelin, Bashar Nuseibeh, Erich Prem, and Allison Stanger (Eds.). 2024.

 
 Introduction to Digital Humanism: A Textbook .

 
 Springer Nature Switzerland, Cham.

 
 

 https://doi.org/10.1007/978-3-031-45304-5 

 

 
 Wexler et al . (2019) 
 
James Wexler, Mahima Pushkarna, Tolga Bolukbasi, Martin Wattenberg, Fernanda Viegas, and Jimbo Wilson. 2019.

 
 The What-If Tool: Interactive Probing of Machine Learning Models.

 
 IEEE Transactions on Visualization and Computer Graphics PP (08 2019), 1–1.

 
 
 https://doi.org/10.1109/TVCG.2019.2934619 

 

 
 Woodworth et al . (2017) 
 
Blake Woodworth, Suriya Gunasekar, Mesrob I. Ohannessian, and Nathan Srebro. 2017.

 
 Learning Non-Discriminatory Predictors.

 
 
 
 arXiv:1702.06081 [cs.LG]

 

 
 Wu et al . (2021) 
 
Chuhan Wu, Fangzhao Wu, Xiting Wang, Yongfeng Huang, and Xing Xie. 2021.

 
 FairRec: Fairness-aware News Recommendation with Decomposed Adversarial Learning.

 
 
 
 arXiv:2006.16742 [cs.IR]

 

 
 Wu et al . (2018) 
 
Yongkai Wu, Lu Zhang, and Xintao Wu. 2018.

 
 Fairness-aware Classification: Criterion, Convexity, and Bounds.

 
 
 
 arXiv:1809.04737 [cs.LG]

 

 
 Xu et al . (2020) 
 
Depeng Xu, Wei Du, and Xintao Wu. 2020.

 
 Removing Disparate Impact of Differentially Private Stochastic Gradient Descent on Model Accuracy.

 
 
 
 arXiv:2003.03699 [cs.LG]

 

 
 Xu et al . (2021) 
 
Han Xu, Xiaorui Liu, Yaxin Li, Anil K. Jain, and Jiliang Tang. 2021.

 
 To be Robust or to be Fair: Towards Fairness in Adversarial Training.

 
 
 
 arXiv:2010.06121 [cs.LG]

 

 
 Xu et al . (2023) 
 
Ziqi Xu, Jixue Liu, Debo Cheng, Jiuyong Li, Lin Liu, and Ke Wang. 2023.

 
 Disentangled Representation with Causal Constraints for Counterfactual Fairness. In Advances in Knowledge Discovery and Data Mining , Hisashi Kashima, Tsuyoshi Ide, and Wen-Chih Peng (Eds.). Springer Nature Switzerland, Cham, 471–482.

 
 

 

 
 Yang et al . (2023a) 
 
Jenny Yang, Andrew Soltan, David Eyre, and David Clifton. 2023a.

 
 Algorithmic fairness and bias mitigation for clinical machine learning with deep reinforcement learning.

 
 Nature Machine Intelligence 5 (07 2023), 1–11.

 
 
 https://doi.org/10.1038/s42256-023-00697-3 

 

 
 Yang et al . (2023b) 
 
Mengyue Yang, Jun Wang, and Jean-Francois Ton. 2023b.

 
 Rectifying Unfairness in Recommendation Feedback Loop. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval ( conf-loc , city Taipei /city , country Taiwan /country , /conf-loc ) (SIGIR ’23) . Association for Computing Machinery, New York, NY, USA, 28–37.

 
 

 https://doi.org/10.1145/3539618.3591754 

 

 
 Yogarajan et al . (2022) 
 
Vithya Yogarajan, Gillian Dobbie, Sharon Leitch, Te Taka Keegan, Joshua Bensemann, Michael Witbrock, Varsha Asrani, and David Reith. 2022.

 
 Data and model bias in artificial intelligence for healthcare applications in New Zealand.

 
 Frontiers in Computer Science 4 (2022), N/A.

 
 

 https://www.frontiersin.org/articles/10.3389/fcomp.2022.1070493 

 

 
 Yu et al . (2022) 
 
Eric Yang Yu, Zhizhen Qin, Min Kyung Lee, and Sicun Gao. 2022.

 
 Policy Optimization with Advantage Regularization for Long-Term Fairness in Decision Systems.

 
 
 
 arXiv:2210.12546 [cs.LG]

 

 
 Zafar et al . (2017a) 
 
Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, and Krishna P. Gummadi. 2017a.

 
 Fairness Beyond Disparate Treatment Disparate Impact: Learning Classification without Disparate Mistreatment. In Proceedings of the 26th International Conference on World Wide Web (WWW ’17) . International World Wide Web Conferences Steering Committee, Australia, N/A.

 
 
 https://doi.org/10.1145/3038912.3052660 

 

 
 Zafar et al . (2017b) 
 
Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, and Krishna P. Gummadi. 2017b.

 
 Fairness Constraints: Mechanisms for Fair Classification.

 
 
 
 arXiv:1507.05259 [stat.ML]

 

 
 Zhang et al . (2018) 
 
Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. 2018.

 
 Mitigating Unwanted Biases with Adversarial Learning.

 
 
 
 arXiv:1801.07593 [cs.LG]

 

 
 Zhang and Wang (2021) 
 
Dell Zhang and Jun Wang. 2021.

 
 Recommendation Fairness: From Static to Dynamic.

 
 
 
 arXiv:2109.03150 [cs.IR]

 

 
 Zhang et al . (2023c) 
 
H. Zhang, L. Wang, Y. Sheng, X. Xu, J. Mankoff, and A.K. Dey. 2023c.

 
 A Framework for Designing Fair Ubiquitous Computing Systems. In Adjunct Proceedings of the 2023 ACM International Joint Conference on Pervasive and Ubiquitous Computing the 2023 ACM International Symposium on Wearable Computing . Association for Computing Machinery, Mexico, 366–373.

 
 

 https://doi.org/10.1145/3594739.3610677 

 

 
 Zhang et al . (2023b) 
 
KeXuan Zhang, QiYu Sun, ChaoQiang Zhao, and Yang Tang. 2023b.

 
 Causal reasoning in typical computer vision tasks.

 
 Science China Technological Sciences N/A, N/A (2023), 1–16.

 
 
 

 
 Zhang et al . (2023a) 
 
Wenbin Zhang, Tina Hernandez-Boussard, and Jeremy Weiss. 2023a.

 
 Censored Fairness through Awareness.

 
 Proceedings of the AAAI Conference on Artificial Intelligence 37, 12 (Jun. 2023), 14611–14619.

 
 
 https://doi.org/10.1609/aaai.v37i12.26708 

 

 
 Zliobaite (2015) 
 
Indre Zliobaite. 2015.

 
 On the relation between accuracy and fairness in binary classification.

 
 
 
 arXiv:1505.05723 [cs.LG]

 

 
 Úrsula Hébert-Johnson et al . (2018) 
 
Úrsula Hébert-Johnson, Michael P. Kim, Omer Reingold, and Guy N. Rothblum. 2018.

 
 Calibration for the (Computationally-Identifiable) Masses.

 
 
 
 arXiv:1711.08513 [cs.LG]

 

 
 
 
 

## Appendix A Appendices

 

### A.1. Fair AI Solutions

 
 Designing solutions for biased models, cater for several concepts like responsible AI, explainability, transparency, accountability and interpretability ( Cheng et al., 2021 ) . Regardless of the distinct ethical issues they tackle, they all share a unified aim of developing "Fair AI" ( Richardson and Gilbert, 2021 ) . As mentioned before, automated tools are being utilized increasingly in making critical decisions surrounding individuals and as such raises concerns about any bias present. The sensitivity of this issue has propelled researchers to come up with solutions for it. Currently, there are several software toolkits available for achieving "Fair AI". We are going to talk about the most popular ones.

 
 

#### A.1.1. 

 
 IBM’s AI Fairness 360 ( Bellamy et al., 2019 ) 
 
 AIF360 is an open-source python toolkit that addresses concerns surrounding bias in AI models. The aim of the developers include identifying and quantifying potential bias in datasets used to train models, examining the sources of bias and employing various techniques to mitigate the bias. They utilize a thorough set of fairness metrics including (but not limited to) statistical parity and equalized odds. They also provide extensive explanations for these fairness metrics. Their bias mitigation algorithms include techniques like adversarial de-biasing, group fairness optimization and more. The framework itself is flexible and lets users integrate their own algorithms. Moreover it provides a user-friendly interface and a testing infrastructure to ensure code is reliable. In general, this tool is a valuable addition to the solution space of fair AI.

 
 
 

#### A.1.2. LinkedIn’s Fairness Toolkit LiFT ( Vasudevan and Kenthapadi, 2020 ) 

 
 LiFT is developed to deal with the issue of fairness and bias in large ML models. The main idea of this tool is to detect and mitigate bias in models that are used for making critical decisions. One of the main advantages of this toolkit is that it leverages Apache Spark to effectively manage large datasets. Like AIF360, LiFT also caters for various fairness metrics including (but not limited to) statistical parity and equalized odds. It investigates training data and detects bias based on sensitive attributes. It utilizes various mitigation strategies including post-processing to reduce bias and promote fairness. It provides flexibility with integrating this into the different stages of the ML pipeline. Additionally the APIs for LiFT, are user-friendly and easy to implement. All in all, this toolkit is a great option that ensures fairness in applications, is scalable and provides accountability and interpretability.

 
 
 

#### A.1.3. Google’s What-If Toolkit WIT ( Wexler et al., 2019 ) 

 
 This tool strives to address the two main challenges of evaluating an ML model: the difficulty in interpreting how the model produces an outcome and the lack of diverse testing (using inclusive dataset and hypothetical scenarios). This tool lets the users modify inputs and investigate how that changes the outcome, essentially giving the opportunity to explore hypothetical scenarios. Users can also explore the model’s behavior by utilizing feature importance analysis and visualizations. A perk of this tool is that it requires minimal coding so it proves to be helpful not only to people with technical background but also to those without. This tool enables users to acquire valuable insight about the model being used, and in turn promotes transparency, reliability and interpretability.
 
 
 The authors in ( Richardson and Gilbert, 2021 ) extensively discusses about fairness solutions in practice for AI bias, . This paper can provide more in-dept knowledge in this domain.

 
 
 
 

### A.2. Paper selection Process for Graph

 
 The first step to this process was to use the query string:
 
 fair OR fairness OR bias AND ( ai OR artificial AND intelligence OR machine AND learning OR ml ) AND PUBYEAR 2016 AND PUBYEAR 2024 AND ( LIMIT-TO ( LANGUAGE , "English" ) ) AND ( LIMIT-TO ( DOCTYPE , "ar" ) OR LIMIT-TO ( DOCTYPE , "cp" ) OR LIMIT-TO ( DOCTYPE , "ch" ) OR LIMIT-TO ( DOCTYPE , "bk" ) ) AND ( LIMIT-TO ( EXACTKEYWORD , "Machine Learning" ) OR LIMIT-TO ( EXACTKEYWORD , "Artificial Intelligence" ) ) 
 
 in the advanced search section of scopus. The papers were then exported as csv and categorized using the GPT-extension (in Google Sheets) to ensure they are indeed related to the domain of fairness, bias and/or XAI. Then the number of papers were plotted against the years.