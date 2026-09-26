Causality from Bottom to Top: A Survey 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY-SA 4.0
 
 
arXiv:2403.11219v1 [cs.AI] 17 Mar 2024 
 
 

# Causality from Bottom to Top: A Survey

 
 
 Abraham Itzhak Weinberg
 
 Affiliation:  AI-WEINBERG, AI Experts, Tel Aviv, Israel, aviw2010@gmail.com
 
    
 Cristiano Premebida
 
 Affiliation:  University of Coimbra, Dept of Electrical and Computer Engineering, Institute of Systems and Robotics, Coimbra, Portugal
 
    
 Diego Resende Faria
 
 Affiliation:  School of Physics, Engineering and Computer Science, University of Hertfordshire, Hatfield, Hertfordshire AL10 9AB, U.K.
 

 Abstract 
 
 Causality has become a fundamental approach for explaining the relationships between events, phenomena, and outcomes in various fields of study. It has invaded various fields and applications, such as medicine, healthcare, economics, finance, fraud detection, cybersecurity, education, public policy, recommender systems, anomaly detection, robotics, control, sociology, marketing, and advertising. In this paper, we survey its development over the past five decades, shedding light on the differences between causality and other approaches, as well as the preconditions for using it. Furthermore, the paper illustrates how causality interacts with new approaches such as Artificial Intelligence (AI), Generative AI (GAI), Machine and Deep Learning, Reinforcement Learning (RL), and Fuzzy Logic. We study the impact of causality on various fields, its contribution, and its interaction with state-of-the-art approaches. Additionally, the paper exemplifies the trustworthiness and explainability of causality models. We offer several ways to evaluate causality models and discuss future directions.

 
 
 
 Keywords: Causality, Aritificial Intelligence (AI), Machine Learning (ML), Explainable Aritificial Intelligence (XAI), Big data, Reinforcement Learning (RL), Generative AI (GAI), Fuzzy Logic

 
 

## 1 Introduction

 
 Causality is one of the fundamental ways to explain phenomena. It is used by human from the dawn of history as a way for explaining results, behaviors and other facts.
Causality can be defined as a relationship between an event (called the cause) and a second event (called the effect), where the cause brings about the effect or directly influences its occurrence [ 1 ] .
In addition, the intuitive nature of causality makes it a common method for young children to explain the reasons behind an effect. 
 Causality can be divided hierarchy into layers or rungs such as seeing, doing and imagining [ 2 , 3 , 4 ] . These are related respectively to association, intervention, and counterfactuals. Each level can be differentiated by its activities and the answers it can provide to relevant question [ 5 ] . The activities and questions are added to the precedence level.
There are several key characteristics of causality that make it a popular concept. It explains relationships between phenomena, providing a framework for understanding why one event or thing leads to another [ 6 ] . Looking for causes helps make sense of the world. Causality also allows for prediction and control, as understanding the cause of something enables us to potentially predict its future occurrences and manipulate causal factors [ 7 ] . Furthermore, causality satisfies a fundamental psychological need, as humans innately seek order, purpose, and patterns in the complex world around us [ 6 ] . It provides a clear connection between events, meeting the human need for explanation and comprehension. Moreover, it facilitates learning and decision making by enhancing knowledge and enabling better-informed choices [ 8 ] . By understanding causes, individuals can learn from past experiences to avoid negative consequences or replicate positive outcomes. Causality aligns with human common sense, as it corresponds well to everyday observations of physical and natural processes. Additionally, causality is scientifically useful as it enables scientists to systematically investigate phenomena, form testable hypotheses, and advance fundamental theories. Causality’s integral role in the scientific method further underscores its significance in scientific inquiry. In summary, causality is popular because it provides structure, predictability, understanding, and a sense of control, all of which are compelling and psychologically rewarding for human minds.
 Throughout history, the concept of causality has been explored and developed by various philosophers and thinkers. In ancient Greece, philosophers such as Plato and Aristotle distinguished between formal and material causes [ 9 ] , while Hellenistic thinkers delved into the realms of chance versus determinism [ 10 ] . In the Middle Ages, scholars built upon Aristotelian foundations to investigate causation in relation to philosophy, theology, and physics [ 11 ] . The early modern period saw Francis Bacon proposing an empiricist theory of causality based on regular succession [ 12 , 13 ] , and Rene Descartes articulating deterministic causation through laws of nature [ 14 ] . 
 The 18th century brought David Hume’s argument that human perception is limited to observing constant conjunction rather than necessary connection between causes and effects [ 15 ] , and Immanuel Kant introduced transcendental idealism in relation to causality [ 16 ] . The 19th century witnessed advancements in statistics and probabilistic reasoning, with John Stuart Mill developing a “Boolean canon of causation for systematic investigation of causal relationships” [ 17 ] . In the early 20th century, Bertrand Russell analyzed causal propositions, and the advent of quantum mechanics challenged determinism, sparking debates [ 18 ] . Carl Hempel’s deductive-nomological model of explanation and the formalization of probabilistic causation using conditional independence through Bayesian networks emerged in the late 20th century [ 19 ] . 
 In the modern era, causality has become integral to fields such as Machine Learning (ML), economics, and statistics. New frameworks for discovery from data, including interventions, counterfactuals, and causal calculus, have been introduced.
Key developments include the formulation of causality using probability theory and graphical models, such as Bayesian networks, by researchers like Pearl [ 20 ] and Spirtes [ 21 , 22 ] . Some additional influential researchers among others, in the field of causality include Robins and Rubin [ 23 ] , Neyman [ 24 ] , Zhang [ 25 ] Gelman [ 26 ] , Mooji [ 27 ] , Athey, Imbens [ 28 ] , Card [ 29 ] , and Angrist [ 30 ] .
 Throughout modern times, the field of causality has experienced significant developments, particularly in its relationship with ML. Here is a rough timeline highlighting some major milestones, as shown in Figure  1 :

 
 
 Figure 1: Causality timeline with the key milestones over the last 50 years. 
 
 
 In the 1970s, researchers, including Suppes and others, laid probabilistic foundations for causality by introducing Bayesian networks [ 31 , 32 ] .
Moving into the 1980s, Spirtes et al. developed the PC algorithm (named after
its authors Peter and Clark) [ 25 , 21 , 33 ] , a crucial advancement for discovering causal structures. Additionally, Neyman and Rubin [ 34 , 24 ] formalized the potential outcomes framework, which provided a solid basis for modeling causal effects using counterfactuals.
The 1990s witnessed the progress of causal calculus based on structural models and graphical criteria, pioneered by Pearl and his colleagues [ 35 , 36 ] . This decade also saw the application of Bayesian network classifiers in ML domains, expanding the use of causal inference methodologies [ 37 , 38 ] .
 Advancements continued in the 2000s, with the maturation of constraint-based and score-based structure learning algorithms for causal discovery. These algorithms played a vital role in various fields, including epidemiology, where causal inference methods were widely adopted [ 39 ] . The notions of interventions and counterfactuals have been rigorously established, clarifying assumptions in observational studies [ 40 , 41 ] .
An additional publication in 2000, by Robins et al. [ 42 ] introduced the concept of sensitivity analysis, a technique utilized to assess the robustness of causal inferences to different assumptions or scenarios that may affect the causal relationship. In this article, we will delve into the topic and explore various common approaches and tools for conducting a sensitivity analysis in your study of causal inference.
 In 2006, Petersen et al. [ 43 ] made a significant contribution by introducing a perspective on causal effect estimation in this context. Estimation of direct effects continues to be a crucial aspect of research aimed at understanding mechanistic pathways, including the ways in which exposures lead to the development or prevention of diseases, among other scenarios.
The 2010s marked a period of prolific developments in machine learning interpretability and explainability techniques. Causal explanations utilizing techniques like Shapley values [ 44 ] , Local Interpretable Model-agnostic Explanations (LIME) [ 45 ] , and anchors emerged within the ML community [ 46 , 47 ] . In addition, game-theoretic approaches have provided causal explanations for complex ML models [ 48 ] . It is worth noting that causal explanations aim to explain the behavior of the model, not necessarily the phenomenon that the model describes. The latter can be only concluded if the model of interest allows us to identify the causal query of interest. Moreover, explainability techniques like Shapley-values-based SHAP can lead to misleading results, regardless of whether we aim to explain the model’s behavior or the mechanism behind the modeled phenomenon [ 49 ] .
Causal discovery has been used in biology and genetics [ 50 , 51 , 52 ] as well as industrial applications. Moreover, causal ML found practical applications in domains such as healthcare [ 53 ] and recommendation systems [ 54 , 55 , 56 ] . The development and increase of computational resources enabled using distributed algorithms for causal discovery from massive multidimensional datasets [ 53 ] .
Causal reasoning is also being utilized to study and mitigate unfair biases and discrimination in ML models and algorithms, known as causal fairness [ 57 , 58 , 59 ] .
From 2020 onwards, new frontiers emerged in the field of causality. Causal RL frameworks were introduced, combining RL with causal reasoning [ 60 , 61 , 62 , 63 , 64 ] . Causal RL has introduced causal environment models for offline evaluation, safe exploration, and transfer learning [ 65 , 66 ] .
Causal fairness methods were developed to address and mitigate bias in algorithms [ 67 , 59 , 58 ] . Researchers explored the discovery of causality from the combination of diverse datasets. Additionally, causal environment models were introduced to supplement value functions [ 68 ] .
 Due to the COVID-19 outbreak, there can be found an increase in usage and development of algorithms for causal discovery from observational data, enabling studies in genetics and epidemiology.
Since causality is intuitive it holds the trustworthiness characteristics. This enables causality to support decision makers more than black box AI models and hence to invade more smoothly into organizations and institutions. This can be one of the reasons that usage of causality has expanded over the years and invaded into many applications and market segments. However, there is still a gap between human and AI/ML causality that will be discussed in detail in the following sections.
 According to Pearl [ 69 ] , there is a distinction between causal inference and causality. Causality refers to the philosophical concept of one event or thing (the cause) being responsible for producing another event or thing (the effect), and the nature of causal relationships.
Causal inference is the process of using statistical and computational techniques, experimental, mixed or observational data, and logical reasoning to quantify the strength of causal effects. It aims to determine causal relationships and effects between variables.
There is also a difference between causal inference and causal discovery [ 70 ] . Causal discovery typically involves using data and statistical or computational techniques to retrieve information about causal relationships from the data, while causal inference utilizes the knowledge about causal structure to quantify the strengths of causal relationships and make predictions. Note that this terminology is not always used consistently in the literature.

 
 
 

## 2 Causality Characteristics and Uniqueness

 
 Causality approach is different from other approaches such as ML, statistical correlation and significance, and descriptive methods. Although some of its characteristics can be found in other approaches, their combination and richness are unique in the context of causality.
Some of the important characteristics of causality that distinguish it include Directionality [ 71 ] , Necessity [ 72 ] , Manipulability [ 73 , 74 ] , Asymmetry [ 75 ] , Transitivity [ 76 ] , Invariance [ 77 ] , Explicitness [ 78 ] , Explanation [ 79 ] , Counterfactuals [ 80 ] , and Transportability [ 81 , 82 ] 
In addition causality possesses several unique characteristics that distinguish it from other relationship such as Mechanisms [ 83 ] , Modularity [ 84 ] , Interventions [ 85 ] , Discrimination and Attribution [ 86 , 81 ] as can be seen in Figure  2 :.
 

 
 
 Figure 2: Main characteristics, as recognized by the relevant literature, that makes Causality distinguishable of other AI domains. 
 
 
 The phrase “correlation does not imply causation” is a well-known concept in research and statistics [ 87 ] . It highlights the important distinction between correlation and causation [ 88 ] . One key factor that sets causality apart from correlation is directionality. Causal relationships involve a clear sense of direction, indicating that the cause precedes the effect [ 71 ] .
In contrast, correlations for instance, lack inherent directionality and can go either way.
Another distinguishing characteristic are sufficiency and necessity [ 89 , 90 , 72 ] . Causality implies that a cause is either sufficient, necessary, or both for its effect to occur. Correlations, on the other hand, do not imply necessity or sufficiency and can exist without a cause-effect relationship.
Manipulability is a defining feature of causal relationships [ 73 , 74 ] . Causality allows for manipulation or intervention on the cause, resulting in observable changes in the effect. This characteristic provides researchers with the ability to test and examine causal relationships more directly.
 Asymmetry is a fundamental distinction between causality and correlations. The causal relationship between two variables, such as ‘A’ causing ‘B’, is distinct and asymmetric from the reverse relationship of ‘B’ causing ‘A’ [ 75 ] . In contrast, correlations exhibit symmetry, measuring the statistical relationship between variables without implying causation.
Transitivity, another property of causality, allows for chaining causal relationships. If ‘A’ causes ‘B’ and ‘B’ causes ‘C’, then ‘A’ is considered an (indirect) cause of ‘C’. This concept of transitivity provides a deeper understanding of the interconnectedness of causal relationships [ 76 ] . Correlations, on the other hand, do not inherently possess transitivity.
Invariance is a characteristic unique to causality. Causal mechanisms remain consistent and invariant under different interventions or contexts [ 77 , 83 ] . Assuming the stability of the underlying causal structure, this property allows researchers to make reliable predictions and draw conclusions about causal relationships, whereas correlations can change based on specific circumstances.
Causal theories also emphasize explicitness by making transparent assumptions about underlying mechanisms. This explicitness goes beyond observed correlations and enables a deeper understanding of how and why causal relationships occur [ 78 ] .
 Explanation is an essential aspect of causality. Causality seeks to explain effects in terms of their underlying causes, rather than merely identifying patterns in data. By understanding the causal mechanisms at play, we can gain a more comprehensive understanding of the phenomena being studied [ 79 ] . Causality explanation characteristic increases the trustworthiness. This can be one of the reasons that causality is popular among decision makers.
Counterfactuals play a crucial role in causal reasoning. Causes support reasoning about hypothetical interventions using counterfactuals, allowing researchers to explore the effects of different scenarios [ 80 , 89 , 69 ] . In contrast, correlations do not provide the support for counterfactual reasoning.
Causality allows for transportability of knowledge. Valid causal inferences can extend beyond the specific conditions of observation due to structural or mechanistic assumptions. This property enables the application of causal knowledge to different contexts or populations [ 91 , 82 ] .
 Another characteristic that sets causality apart is the concept of modularity. Complex causal systems can be broken down into autonomous modules, each with defined inputs and outputs [ 84 , 92 ] . By analyzing the compositionality of these modules, researchers can gain insights into the overall causality of the system. This modular approach provides a structured and systematic way of understanding complex causal relationships.
Interventions play a vital role in establishing causation. Through experiments and - under certain circumstances - natural experiments, researchers can isolate and manipulate proposed causal factors while controlling for other variables [ 85 ] . This rigorous approach allows for the discovery of causal relationships through observations made under controlled conditions. Interventions enable researchers to go beyond mere associations and give a platform to falsify causal hypotheses.
Furthermore, causality provides a framework for attribution. It allows us to attribute responsibility, credit, or blame for the occurrence of certain effects. Unlike correlation alone, which merely identifies associations, causality enables us to assign causal responsibility and understand the consequences of specific causal factors [ 81 ] .

 
 
 

## 3 Causality Types, Relationships and Inference

 
 Causality Inference (CI) refers to the process of identifying and understanding causal relationships between variables or events [ 69 ] .
Causal Discovery (CD) involves the analysis and construction of models that depict the inherent relationships within the data, while causal inference seeks to examine the potential effects resulting from altering a specific system [ 93 ] .
It involves determining whether one variable or event directly or indirectly influences the occurrence or outcome of another. We can find several approaches to causality as can be seen in Table 1 .

 
 
 
 
 
 Causality Approaches | 

 
 
 
 
 
 Approach 
 | 
 
 
 Conceptual Framework 
 | 
 
 
 Data Requirements 
 | 
 
 
 Assumptions 
 | 
 
 
 Estimation Methods 
 | 
 
 
 Interpretation 
 | 

 
 
 
 Causal Graphical Models (CGM) [ 94 ] , Hidden Confounding (HC) [ 95 ] 
 | 
 
 
 Graphical representation 
 | 
 
 
 Joint distribution of variables 
 | 
 
 
 Acyclicity, no unmeasured confounders, ignorability (no unobserved confounders) 
 | 
 
 
 Non-parametric identification and estimation 
 | 
 
 
 Direct interpretation based on graphical model structure 
 | 

 
 
 
 Potential Outcome (PO) Framework [ 96 ] 
 | 
 
 
 Counterfactuals 
 | 
 
 
 Potential Outcomes 
 | 
 
 
 Stable Unit Treatment Value Assumption (SUTVA), ignorability (no unobserved confounders) 
 | 
 
 
 Matching, regression, weighting, etc. 
 | 
 
 
 Differences in potential outcomes under different treatment conditions 
 | 

 
 
 
 Difference-in-Differences (DiD) [ 97 ] 
 | 
 
 
 Comparing changes over time 
 | 
 
 
 Treatment and control groups 
 | 
 
 
 Parallel trends between treatment and control 
 | 
 
 
 Regression models with interaction terms, fixed effects 
 | 
 
 
 Average treatment effect based on comparison of changes over time 
 | 

 
 
 
 Instrumental Variables (IV) [ 30 ] 
 | 
 
 
 Exploiting instrumental variables 
 | 
 
 
 Instrumental variables 
 | 
 
 
 Relevance and validity of instrumental variables 
 | 
 
 
 Two-stage least squares, instrumental variable regression 
 | 
 
 
 Causal effects based on association between instrumental variable and treatment 
 | 

 
 
 
 Structural Equation Modeling (SEM) [ 98 , 99 ] 
 | 
 
 
 Modeling structural relationships 
 | 
 
 
 Data for estimating model parameters 
 | 
 
 
 Model-specific assumptions under causal identification 
 | 
 
 
 Maximum likelihood estimation, other SEM-specific methods 
 | 
 
 
 Causal effects based on estimated parameters of the structural model 
 | 

 

 Table 1: Comparison of causal inference approaches, highlighting the conceptual basis, requirements, assumptions, estimation methos, and interpretation capabilities. 
 
 
 Causal Graphical Models (CGM) [ 94 ] use graphical representation and do-calculus, allowing for direct interpretation of causal effects. The Potential Outcome Framework [ 96 ] considers counterfactuals. Difference-in-Differences (DiD) [ 97 ] compares changes over time between treatment and control groups. Instrumental Variables (IV) [ 30 ] exploits instrumental variables to estimate causal effects. Propensity Score Matching (PSM) [ 100 , 101 ] balances covariates and compares outcomes between matched individuals. Structural Equation Modeling (SEM) [ 98 , 99 ] estimates causal effects based on the parameters of a structural model. Each approach has its own conceptual framework, data requirements, assumptions, estimation methods, and interpretation of causal effects.
 Causality encompasses several main types that support understanding the relationships between causes and effects [ 102 , 103 ] . These types shed light on the various ways in which these relationships can manifest. There are some popular types such as Direct or Strong, Indirect or Weak, Necessary, Sufficient, Multiple or Joint, Probabilistic, Common cause, Reverse or Spurious, and causal Homeostasis.
 Direct/strong causality occurs when A directly causes B without any intermediary factors [ 104 ] . An example a pool cue (A) striking a billiard ball directly causes the ball to roll across the felt (B).
Indirect/weak causality occurs when A causes B, but there are intermediary factors involved [ 105 , 106 ] . For instance, poverty indirectly causes poor health outcomes through factors like lack of access to healthcare and nutritious food.
Necessary causality signifies that A is necessary for B to occur, but it may not be sufficient on its own [ 107 ] . For example, lack of oxygen is necessary for death to occur, but other causes can also lead to death.
Sufficient causality indicates that A by itself is sufficient to cause B [ 108 ] . For example, a high enough dose of poisoning may be sufficient to cause death without any other factors involved.
Multiple/joint causality arises when both A and B are required together to cause an effect [ 109 ] . For example, A gene mutation (A) and environmental trigger (B) are necessary for some diseases to manifest.
Probabilistic causality suggests that A increases the probability or risk of B, but it does not guarantee the occurrence of B [ 110 ] . For instance, obesity increases the risk of diabetes, but not everyone who is obese will necessarily develop diabetes.
 Common cause causality occurs when a third variable, C, causes both A and B. For example, genetics C may cause both obesity A and diabetes B.
Reverse/spurious causality emphasizes that the assumed cause A may actually be caused by or correlate with the effect B, rather than vice versa [ 111 ] . It is important to correctly identify causes and effects to avoid misinterpretation.
Additionally, there is the concept of causal homeostasis. Causal homeostasis refers to the tendency of causal systems to resist change and maintain stability, even when external interventions or perturbations occur [ 112 , 113 , 114 ] . This stability is achieved through feedback loops and networked causal mechanisms within systems that absorb, redistribute, or redirect incoming influences. As a result, interventions may have more limited effects than expected, as homeostatic regulatory processes work to return causal dynamics to their baseline state. Understanding causal homeostasis provides insights into why observational and interventional distributions, even when standard causal assumptions are violated.
It is important to recognize that causality is often probabilistic rather than definite, with various factors and complexities at play in cause-effect relationships. 
 Various types of causal relationships exist, each with distinct characteristics and implications.
Confounding relationships can arise when third variables influence both the exposure and outcome. For example, stress, diet and exercise habits are interrelated - higher stress can reduce healthy behaviors but poor diet and exercise can also increase stress levels through physiological pathways. At the same time, all three influence health outcomes like heart disease risk. Attempting to isolate any direct causal effect, such as between stress and heart disease, is complicated by the bidirectional links between stress, diet, exercise and their collective impacts on health [ 106 ] . Spurious relationships, on the other hand, are correlations that seem causal but are actually due to confounding factors [ 115 ] . Mediating relationships occur when the effect of one variable on another is partially transmitted through a mediator [ 116 ] , such as smoking cessation programs reducing lung cancer by reducing smoking. Moderating relationships occur when the strength or direction of a causal effect depends on a third variable, such as social support moderating the effect of stress on mental health [ 117 ] . Lastly, bidirectional/recursive relationships involve two variables causally influencing each other in an ongoing feedback loop over time [ 118 ] , for example, optimism and life success reinforcing each other. To accurately distinguish these different types of relationships from observational data alone, it is crucial to identify the causal graphical structure and apply appropriate causal inference techniques.
 As shown in Figure  3 , we offer a taxonomy based on the following classes: Direction, Necessity, Relationship Type, Evidence Strength, Number of Causes, Temporal Sequence, Mechanism.

 
 
 Figure 3: Causality Taxonomy based on seven representative categories: mechanism, direction, necessity, relationship, evidence, causes, and temporality. 
 
 
 Direction describes whether causes happen before, after, or simultaneously with their effects. Necessity distinguishes causal factors by their level of necessity or sufficiency to produce an effect. Relationship Type characterizes the link between cause and effect as direct, indirect, or spuriously associated. Evidence Strength categorizes the conclusiveness of empirical evidence supporting a causal claim. Number of Causes addresses whether one or multiple factors jointly contributed to bringing about an effect. Temporal Sequence considers the timing of causes and effects as simultaneous, cause preceding effect. Mechanism refers to the underlying processes or means through which causation is achieved, such as physical, probabilistic, and intentional.

 
 
 

## 4 Causality Applications and Usages

 
 Causality plays a crucial role in various fields, impacting numerous applications. In medicine and healthcare, understanding disease etiology, evaluating treatment effectiveness, and precision medicine are important applications. Additionally, causality aids in clinical decision support, predictive risk modeling, and epidemiology [ 119 ] . 
 Transportability is a crucial aspect where causal models are needed to generalize across different contexts and populations [ 120 , 121 ] . This is particularly relevant in healthcare, where diverse groups may require tailored interventions and treatments. Heterogeneity is another key consideration, as causal understanding helps identify the treatments that work best for specific individuals. Precision medicine heavily relies on causal analysis to determine personalized approaches to healthcare [ 122 ] . Experimentation is an area where causal modeling excels, enabling the optimization of experimental design when fully randomized experiments are either unethical or impractical to conduct. Time series analysis benefits greatly from causal modeling as it allows for the examination of dynamic processes unfolding over time [ 123 ] . This is essential for accurate forecasting in various applications.
 Moving to the realm of economics and finance, causality is essential for assessing the effects of policies and regulations, predicting the impact of economic decisions, and understanding consumer choice factors [ 124 ] . It also plays a role in forecasting economic trends [ 125 ] .
Mechanism design utilizes causal models to develop incentives and policies that shape desired outcomes. This application is commonly seen in the field of economics [ 126 ] . Fairness becomes an important consideration in algorithmic decision-making, and causal reasoning plays a significant role in identifying unfair biases and understanding their root causes [ 127 ] . The ability to answer what-if questions about interventions is critical for decision support [ 2 ] . Causal models allow for the assessment of the potential impacts of different interventions, aiding in informed decision-making.
In the field of education, causality helps in determining education interventions and reforms, implementing personalized learning approaches, and assessing teaching methods [ 128 ] . It aids in understanding attrition factors and enables predictive analytics for educational outcomes. Public policy benefits from causality by evaluating the effectiveness of social and environmental programs, assessing the impact of laws and bills, and facilitating evidence-based decision making [ 129 , 93 ] . 
 Causality is also significant in recommender systems [ 130 , 54 ] , allowing for the explanation of recommendations, counterfactual reasoning for fairness, and estimating the effects of exposure to certain items. In fraud and anomaly detection [ 131 ] , causality helps in isolating causative factors for abnormalities rather than relying solely on correlations. It also contributes to predictive maintenance strategies. 
 In the field of robotics and control [ 132 , 133 ] , causality is crucial for causal world modeling, decision making under interventions, and simulating the downstream effects of actions. Sociology benefits from causality by modeling influence and diffusion networks, understanding shifts in group behavior, and exploring social determinants of various outcomes [ 134 ] .
 In marketing and advertising, causality aids in estimating campaign effectiveness, targeting high-potential customer segments, and optimizing acquisition and retention strategies [ 135 ] . In the domain of cybersecurity, causality plays a role in predicting vulnerabilities and risks, tracing the root cause of intrusions and attacks, and evaluating alternative security control measures [ 136 ] .
Explainability is a major advantage of causal models as they provide explanations in terms of cause-effect relationships rather than mere correlations. This enhances the transparency and interpretability of AI systems [ 137 ] . 
 Counterfactual analysis, which involves generating realistic alternatives, is valuable for assessing the impacts of decisions that were not taken. This helps in understanding the potential outcomes of different choices.
Complex systems often require combinatorial models that involve causal reasoning across different formalisms. This interdisciplinary approach allows for a comprehensive understanding of intricate systems. In summary, causality plays a critical role in facilitating transfer learning, ethical and fair decision-making, policy optimization, and the development of transparent, explainable, and trustworthy AI systems [ 138 ] .
Causality is a fundamental concept that finds applications in various fields. Science relies on causality to establish cause-and-effect relationships between variables. 
 In philosophy, causality is a central concept in metaphysics and epistemology [ 139 ] . The legal field employs causality to determine liability and responsibility for damages or injuries [ 140 ] . In medicine, causality is crucial for understanding disease causes and developing treatments [ 119 ] . In the social sciences, causality is employed to understand social phenomena such as poverty, crime, and inequality [ 141 ] . 
 Business leaders utilize causality to understand the causes of success and failure, develop strategies for improving performance, and drive customer behavior. Economics employs causality to understand relationships between economic variables. Environmental science employs causality to understand the causes of environmental phenomena.
Overall, causality is a fundamental concept used across various fields. It helps understand relationships between variables, establish cause-and-effect connections, and develop interventions and strategies to address challenges in different domains.

 
 
 

## 5 Preconditions and Architecture for Implementing Causality on Datasets

 
 Establishing causality from datasets is a complex task that requires careful consideration. In ML, defining the target variable and selecting relevant features are crucial steps in the implementation process. The choice of ML models plays a significant role, as they can be trained to improve the chances of obtaining the desired results. Nonetheless, when aiming to establish causality from datasets, it is essential to acknowledge and address several key preconditions. These preconditions, which help ensure the validity and reliability of causal inferences, involve considerations such as temporal order, confounding factors, and the presence of a plausible mechanism [ 142 ] .
 The database architecture also influences the causal inference process. Graph databases, due to their ability to represent Directed Acyclic Graphs (DAGs), provide a suitable platform for causal analysis. The DAG property of graph databases enables faster causal inference compared to popular relational databases. Additionally, graph databases can be effectively used to represent and analyze causality [ 143 ] .
 The causal relationships can be effectively represented in a graph database by modeling them as directed edges or links between nodes, allowing for the explicit depiction of connections between objects. This approach enables the tracing of causal pathways and influences, empowering the analysis of connectivity and propagation within complex systems through graph queries and algorithms. By identifying interconnected causal networks and understanding how causal power propagates, it becomes possible to determine all entities indirectly impacted by a specific event or condition. In addition, ML techniques such as network embedding can be employed to infer unknown causal links by analyzing patterns of connectivity within a causal graphs. Graph databases also enable the simulation of what if scenarios, as they allow for the modification of causal link attributes or the selective addition or removal of nodes and edges.
 This facilitates the exploration of how a causal network may behave under varying conditions or interventions. Moreover, the contextualization of causality is supported by assigning properties to nodes and edges in the graph, representing contextual factors that influence or modify causal relationships. This allows for the modeling of more nuanced and conditional forms of causality. Graph databases also provide a unified structure that facilitates the integration of diverse causal data types, such as experimental, observational, temporal, and spatial data, into a single queryable network, offering a comprehensive approach to studying causality. The Graph database architecture provides an optimal architecture that makes the causal inference process more efficient.
 As mentioned above, there are inherent properties that are essential for casual inference. Firstly, temporal precedence is crucial, meaning that the potential cause must occur before the alleged effect, establishing a clear temporal order. Secondly, there needs to be a statistical association or correlation between the potential cause and effect variables in the data. Additionally, it is important to rule out confounding variables, alternative explanations, or common causes for the relationship. This can be achieved through statistical control. Furthermore, a plausible mechanism should be understood, based on scientific theory or knowledge, to explain how the cause could lead to the effect. It is essential to note that correlation does not imply causation [ 144 ] .
 The gold standard for establishing causality is experimental data, where the cause is intentionally manipulated to confirm its impact on the effect. Observational data, while valuable, is considered weaker. Another important factor is the presence of a dose-response relationship [ 145 ] , wherein larger or smaller doses of the cause correspond to larger or smaller magnitudes of the response. Consistency across studies, contexts, and populations strengthens the evidence of causality, as it replicates the relationship.
 Specificity is another precondition, indicating that a singular cause should lead to a specific effect and not a broad range of variables. Overly broad conclusions weaken the strength of the causal relationship [ 146 ] .
Establishing causality requires carefully designed studies, appropriate statistical techniques, and the elimination of threats to validity. Merely possessing large, correlated datasets is insufficient for determining causation. 
 Additional preconditions for establishing causality from datasets include considering sample size. Larger sample sizes are required to reliably detect even modest causal effects, control for confounding variables, and replicate findings [ 147 ] . Insufficiently powered studies cannot rule out alternative explanations. Data quality is also critical, necessitating accurate, complete, and multi-variate data with minimal missing information. Measurement error, selection bias, and attrition can undermine causal conclusions.
Counterfactual analysis is vital, as it involves considering what would have occurred in the absence of the supposed cause. This is challenging without experimental manipulation of potential causes. Observational studies require stronger assumptions.
As we discussed before about the Graph database, DAGs can assist in formalizing assumptions about causal structure, identifying confounding variables, and determining if the data and research design can establish causality between specific variables.
Replication is pivotal, as causal conclusions gain strength when multiple independent studies using different datasets and methods yield consistent effects, minimizing the likelihood of chance findings [ 148 ] .
Theoretical consistency is another important factor, as causal hypotheses must align with scientific theory. Anomalous findings should be scrutinized carefully, as they may not reflect true causality [ 149 ] .
Establishing causality from observational data requires assembling converging evidence from multiple angles and addressing threats to validity. It is a gradual and multifaceted process, rather than an absolute conclusion.

 
 
 

## 6 Causality Evaluation and Metrics

 
 One of the most important ways to relate to the extracted results is to use set of criteria and evaluation metrics.
No single metric quantifies causal understanding; rather a combination provides a holistic evaluation [ 150 ] . The goal metrics focus on are also application dependent.
The previous section mentioned the essential datasets characteristics that enable casual inference. In this section, we consider the way to evaluate the extracted causality model. Before diving into the key aspects and metrics, there is a need to emphasise some important datasets characteristics that are support the model evaluation. The ideal dataset has to include both observational and interventional/experimental data. In addition, causality evaluation has to be based on on multiple relevant datasets in order to test generalization.
 In order to assess causal models metrics such as effect sizes, causal discovery accuracy, and counterfactual quality should be considered.
To establish a baseline for comparison, it is recommended to compare causal models to non-causal ML models. It is also valuable to evaluate them against causal inference methods like potential outcomes and matching. Another important factor is sample complexity. Testing how data requirements scale up for increasingly complex and unidentified causal problems helps understand the limitations and scalability of the models.
Assessing the robustness of results to violations of modeling assumptions is another critical aspect. Analyzing how sensitive causal inferences are to choices in the model, metric, hyperparameters, and training data through sensitivity analysis is necessary. Furthermore, evaluating the interpretability of models is essential to determine if they provide intelligible explanations for predictions and facilitate scientific insights.
 Models should also be evaluated regarding how well they convey uncertainty about their causal conclusions. Benchmarking the transportability of models’ causal conclusions to new domains and populations is crucial to understand their applicability beyond the training data. For dynamic or time series problems, it is important to test the models’ ability to discern short-term versus long-term effects.
Using feedback is also recommended in the model evaluation process. For more complex problems with interactive and bidirectional causal relationships can provide insights into the models’ performance in real-world scenarios. Thorough evaluation requires testing across multiple dimensions on standardized benchmarks using both simulated and real-world data. It is important to note that no single study can definitively validate a causal approach.
 Several common metrics can be used to evaluate causal models and causal inference. Some of them are relevant to models such as ML while others are tailored to causality. These metrics include prediction error, intervention accuracy, effect estimate accuracy, confidence calibration, ROC AUC, structural Hamming distance, modularity score, causal sufficiency, transportability error, counterfactual sample quality, and counterfactual effect consistency. Each of these metrics offers a different perspective on the models’ effectiveness in capturing causal relationships and estimating causal effects.
 In addition to these evaluation methods, Hill’s first criterion can be used to assess the strength of causal connections, specifically the association connection. Hill’s criteria, primarily focused on health and biological perspectives but applicable to other causal associations, include Consistency, Plausibility, Temporality, Dose-response relationship, Experimental evidence, and Mechanistic evidence.
 We already discussed the hierarchy of causal inference and its layers or rungs. By following this hierarchical approach, causality evaluation and metrics help quantify the strength of causal inference and provide valuable insights into the nature of causality.
While there is no universally agreed-upon hierarchy of causal inference methods, some common frameworks discuss causality approaches in a hierarchy or order of strength [ 151 ] . The hierarchy generally progresses from weaker association-based methods using observational data to consistency across studies, plausibility evaluations, establishment of proper temporality and dose-response relationships, and culminates in the strongest evidence from experimentation and mechanistic understanding of the processes involved [ 152 ] . These later approaches help rule out alternate interpretations and provide confirmatory evidence.

 
 
 

## 7 Causality Trustworthiness

 
 Ensuring a model can be trusted is vital for responsible deployment, responsible use, and continued uptake and reliance on the model. It helps address various legal, ethical, and practical concerns regarding AI safety and transparency [ 82 ] . The need for trustworthiness intensifies when the model is extracted without human intervention such as in the case of causality models. As already mentioned above, the causality models by their nature holds the trustworthiness characteristics that can be affirmed by human logic and common sense. The trustworthiness of causality also relates to its inherent ability to explained to its users and will be discussed in the explainablity section in detail.
 When evaluating the trustworthiness of causal claims and models, it is important to consider several key aspects. These aspects provide insights into the reliability and validity of the reported causal conclusions [ 153 ] .
Firstly, data quality plays a critical role, as potential biases, omissions, or errors in the observational data can undermine the accuracy of causal inferences. Additionally, the identifiability of the causal structure and assumptions from the given data is crucial, as the lack of identifiability can impact the validity of the conclusions.
Confounding is another vital aspect to address, ensuring that all common causes of the treatment and outcome variables are adequately measured and accounted for to avoid biased estimates. The appropriateness of the assumed causal graph structure and model form, known as model specification, should also be carefully examined, allowing flexibility to validate assumptions.
 Accurate parameter estimation is essential to determine the precision and accuracy of the estimated causal effects from the data, while being cautious of the potential for overfitting. Transportability is another consideration, ensuring that the estimates can be generalized beyond the observed context and are not contingent on unmeasured factors. 
 Sensitivity analysis provides an understanding of the robustness of the conclusions to violations of assumptions through what-if analysis. Predictive performance evaluation focuses on how well the estimated effects can predict the outcomes of planned interventions.
Transparency is a crucial aspect of establishing trust, requiring clear specification of assumptions, limitations, and methodological details to enable scrutiny. Lastly, evaluation should go beyond accuracy metrics and consider the suitability of the conclusions for intended uses, incorporating valuable stakeholder feedback.

 
 
 

## 8 Causality and Explainable AI (XAI)

 
 Explainable AI (XAI) refers to the research and methods aimed at providing transparency and understandability to the decision-making process of artificial intelligence models [ 154 ] . It addresses the challenge of interpreting the outcomes and inner workings of complex AI systems, which are often perceived as black boxes. Causality, on the other hand, can serve as a surrogate model for these black box AI models, enhancing their level of explainability. By combining causality and XAI, a more comprehensive understanding of AI decision-making can be achieved.
Causality and Explainable AI (XAI) are two interconnected fields that have garnered significant attention in recent years. Causality serves as a surrogate model for AI black box models, augmenting their level of explainability. By combining causality and XAI, a more comprehensive understanding of AI decision-making can be achieved [ 155 ] .
 One aspect where causality and XAI intersect is in providing causal explanations. XAI endeavors to shed light on the decision-making process of AI models, while causality helps identify the causal relationships between variables that contribute to these decisions [ 156 ] . By merging these disciplines, it becomes possible to generate explanations that not only describe the contributing factors but also elucidate the causal connections between them.
 Another area of synergy lies in causal attribution. Causal attribution involves assigning credit to the factors that influence a decision or outcome [ 155 ] . XAI can pinpoint the most significant contributors to an AI model’s decision, while causality can determine the causal relationships between these factors. Integrating causal attribution and XAI yields a more comprehensive understanding of how AI models arrive at their decisions.
 Causal visualization is yet another domain where causality and XAI converge. Causal visualization entails creating visual representations of the causal relationships between variables [ 157 ] . XAI can generate visual explanations of AI decision-making processes, while causal visualization captures the causal links between the contributing factors. By combining these approaches, more informative and intuitive explanations of AI decision-making can be crafted.
Moreover, the combination of counterfactual explanations and XAI offers valuable insights into AI models’ decision-making processes [ 158 ] . Counterfactual explanations explore hypothetical scenarios by describing what would have transpired had a specific action or event not occurred. XAI can generate such counterfactual explanations, while causality helps identify the causal relationships that contribute to these decisions. This fusion allows for a more comprehensive understanding of AI models’ decision-making.
 Lastly, explainable RL benefits from the integration of XAI and causality [ 159 ] . RL trains AI agents to make decisions based on rewards or penalties. XAI can explain the decisions made by these agents, while causality uncovers the causal relationships between the agents’ actions and the rewards or penalties received. By combining XAI and causality, a more comprehensive explanation of the decision-making process of RL agents can be provided.

 
 
 

## 9 Causality and Machine Learning (ML), AI, Genetic Algorithms (GA), and Generative AI (GAI) Approaches

 
 AI and ML approaches that have been used for solving problems and prediction have their own benefits and drawbacks in comparison to causality and causal inference in particular. This section will shed a light to the differences between the approaches and the interaction as well as combination between them.
 According to Bishop et al. [ 160 ] the integration between AI and causality has its limitations, and simply relying on causal reasoning may not be sufficient to address those limitations. However, a review of the role of causality in developing trustworthy AI systems highlights the potential benefits of incorporating causality [ 82 ] . Causal AI not only has the ability to predict but also offers the capacity to answer questions that traditional ML models cannot. Unlike predictive ML models, Causal ML explicitly accounts for confounders by modeling both the treatment of interest and its impact on the outcome [ 82 ] . This allows Causal ML to isolate the causal impact of treatment on the outcome, going beyond mere correlation.
 There are several key differences between causality-based approaches and traditional ML approaches. Firstly, while causal methods can often utilize observational data, ML typically requires experimental data to establish reliable cause-effect relationships. Secondly, ML focuses on prediction, while causality aims to understand the underlying generative processes and reason about new interventions. Causal graphs make explicit independence assumptions between variables, whereas ML primarily fits functional forms [ 82 ] . Moreover, causal graphs and counterfactual analysis provide human-interpretable explanations of model predictions in terms of causes and effects, whereas ML models are often viewed as black boxes without clear causal interpretations.
 Causality also offers advantages in terms of transportability, as causal inferences may generalize and propagate interventions beyond the training setting, whereas ML models typically do not. Causal structures are generally robust to interventions and conditional distributions, whereas ML models may fail under distribution shift. Additionally, causality helps address issues of fairness, bias, and discrimination by focusing on causal effects rather than associative effects [ 82 ] .
 As mentioned above, in terms of trustworthiness, causal models provide human-interpretable explanations through graphical structures, while ML models are often opaque. Causal assumptions are explicitly stated in causal models, while ML assumptions are implicit in the algorithm and data. Causal effects have the potential to generalize beyond training examples, while ML performance relies heavily on identically distributed test data. Causal graphs encapsulate invariance, while ML predictions can change unpredictably with small input changes. Causal methods can quantify and mitigate unfair biases, whereas ML models risk amplifying biases present in the training data. Causal estimates aim to remain valid under interventions, while ML models often do not transport beyond their training conditions. Lastly, causal conclusions account for violations of assumptions, while ML performance is highly sensitive to violations of Independently and Identically Distributed (IID) assumptions [ 82 ] .
 The Generative AI (GAI) models holds nowadays, an important roles in many fields. Although GAI is a subfield of AI, when comparing causality with GAI the task intensifies. GAI systems and humans exhibit key differences in their understanding of causality. Generative models like Generative Adversarial Networks (GANs) learn patterns and generate new data without a comprehensive causal model of the world, while humans combine data-driven learning with intuitive physical reasoning.
Generative models, such as GANs, excel at learning patterns and generating new data, but they lack a comprehensive causal model of how the world functions. In contrast, humans combine data-driven learning with intuitive physical reasoning.
 AI causality primarily relies on statistics and correlations derived from large datasets, whereas humans infer causality through experimentation, reasoning about mechanisms, and contemplating counterfactuals, such as what-if scenarios.
GAI lacks subjective experience and commonsense knowledge regarding how causal factors like psychology, culture, and societal influences shape human behaviors and decisions.
AI focuses on identifying statistical causes to generate realistic outputs without a profound understanding of how physical, mental, and social factors interact in complex real-world phenomena continuously refine their understanding of causality through lifelong open-ended learning from diverse sources. Generative models, on the other hand, remain confined to their initial narrow training data and predefined objectives.
Human causal attributions involve reasoning about agency, responsibility, and intentions underlying events. In contrast, GAI operates solely on correlations without possessing higher-order faculties for such considerations.
Another type of algorithms, that relates to ML approach are Genetic Algorithms (GA). GA are optimization techniques inspired by biological evolution, aiming to replicate the natural selection process [ 161 ] . Rather than determining causal relationships or assigning causality, they focus on exploring complex search spaces and finding optimal solutions. By applying selection, crossover, and mutation principles to populations of potential solutions, GA iterate through generations and select the fittest candidates to drive the search. While crossover and mutation introduce random changes, mimicking natural genetic variations, genetic algorithms cannot directly infer causal relationships. In ML tasks, such as predicting relationships in data, GA identifies correlations that optimize the fitness function but do not establish causation. Additional analysis of GA results may allow for inferences about potential causal relationships, but determining causality requires controlled laboratory experiments or observational studies. The value of GA solutions lies in their fitness, not necessarily in accurately modeling the true causal structure that generated the data, if causality even applies. GA prioritize optimizing what works rather than explaining why it works causally.

 
 
 

## 10 Causality and Bigdata

 
 Bigdata datasets have been rapidly growing in recent decades due to several technological trends that have increased storage volume, enabled efficient handling by powerful processors, and facilitated the use of advanced query languages and ML models. The bigdata trend encourages institutions and organizations to collect as much data as possible, creating a snowball effect where the more data they collect, the larger data platforms they require. 
 Causality presents significant challenges in the context of bigdata, as causal inference must operate on a large volume of data that is beyond human handling capabilities. Additionally, it needs to accommodate the characteristics of big data known as the 5V’s: velocity, volume, value, variety, and veracity [ 162 ] . This further intensifies the challenge of developing causality models, as it necessitates not only handling vast amounts of data but also dealing with its variety and constructing a causal model within a limited timeframe.
In addition to the above digital interventions, such as A/B testing of user interface modifications and personalized recommendations at scale, offer opportunities to conduct large-scale experiments and uncover cause-effect relationships within real user behavior.
 The understanding of causality in the realm of bigdata analysis involves several key aspects [ 163 ] .
Observational data, which comprises large passively collected datasets, has facilitated the exploration of complex causal structures that were previously too intricate or costly to analyze. However, the quality of inferences heavily relies on the ability to effectively control for confounding factors.
New statistical techniques [ 164 ] , such as graphical models and structure learning algorithms, are being scaled up using methods like subsampling, approximation, and distributed computing. These advancements aim to handle the complexities associated with thousands of variables in causal analysis.
Passive observational studies utilizing extensive patient datasets have emerged as viable alternatives to randomized experiments when identifying treatment effects, thanks to the availability of adequate sample sizes. However, the concern of selection bias still persists.
 The combination of multiple datasets, encompassing diverse sources like electronic health records, social media, and mobile data, can provide complementary insights. On the other hand, addressing challenges related to data alignment and integration is crucial.
Validating predictions and quantifying uncertainties at scale have become feasible due to the ability to re-identify individuals and capture finer-grained attributes. However, this increased capability raises privacy risks that must be carefully managed.
Causal discovery in the context of bigdata poses challenges, particularly in scaling techniques for complete identification of causal structures. Additionally, data integration and the selection of relevant variables remain ongoing obstacles in the pursuit of comprehensive causal analysis.

 
 
 

## 11 Causality and Reinforcement Learning (RL)

 
 Reinforcement Learning (RL) is a branch of ML where an agent learns to make decisions by interacting with an environment. It involves the agent taking actions, receiving feedback in the form of rewards or penalties, and using this feedback to improve its decision-making over time [ 165 ] . RL allows automated agents to learn optimal behavior directly from experience to solve complex real-world challenges through trial-and-error interaction. Its usage is increasingly prevalent nowadays due to emerging of new algorithms and usages such as in Large Language Models (LLM) application as ChatGPT. 
 Causality and RL are connected through their shared goal of decision making through modeling systems and both attempt to address challenges by leveraging observational data. Causality can enhance RL if incorporated appropriately.
Causality RL is an interdisciplinary research field that merges concepts from causal inference and RL to investigate the causal effects of actions in decision-making processes [ 163 ] . Its objective is twofold: to learn a policy that maximizes cumulative rewards and to identify the causal relationships between actions and environmental changes.
 In typical RL scenarios, agents learn to make decisions by interacting with the environment and receiving rewards or penalties [ 166 ] . Nonetheless, in real-world situations, the causal relationships between actions and environmental changes are often unclear, necessitating the agent’s ability to reason about causal effects for informed decision-making.
Causality RL addresses this challenge by integrating causal inference methods into the RL framework [ 167 ] . Agents use causal models to represent the causal relationships between actions and environmental changes, enabling them to infer the causal effects of their actions. This information guides the learning process, ensuring that the agent’s policy maximizes expected rewards while aligning with observed causal data.
 Key concepts in Causality RL include causal graphs, which depict the relationships between variables in the environment, and causal inference, which employs statistical methods and data to infer causal relationships [ 168 ] . Counterfactuals play a role in reasoning about the effects of alternative actions, and causal loops represent situations where the agent’s actions impact the environment, which, in turn, affects the agent’s rewards [ 169 ] .
Causality RL offers several benefits, including improved decision-making by considering causal effects, better handling of confounding variables through inferred causal relationships, and increased transparency through explicit causal models.
To date, challenges persist within Causality RL [ 163 ] . The complexity of causal models can hinder interpretation and learning of causal relationships. Identifying causal relationships may be difficult when they are not readily apparent, and striking a balance between exploration and exploitation poses a challenge, particularly with limited causal information.

 
 
 

## 12 Integration between Causality, Machine and Deep Learning, and Genetic Algorithms

 
 In the previous sections, the difference between causality and ML approaches was mentioned. This section will discuss the interaction and connections between the two. The section relates to Deep Learning (DL) as subfield of ML, unless mentioned otherwise.
Causality and ML are intertwined in several ways, leveraging ML techniques to uncover causal relationships between variables and make predictions based on these relationships. The relationship between causality and ML manifests in various aspects.
Causal inference plays a crucial role, utilizing ML to deduce causal relationships by analyzing patterns in data [ 170 ] . There are approaches of harnessing DL to transform distributions into a representation space such that they are indistinguishable [ 171 ] .
Causal graphs provide a visual representation of causal relationships between variables. ML algorithms facilitate learning the structure of causal graphs from data by identifying patterns of causality among variables [ 163 , 172 ] .
 ML algorithms enable the fitting of causal models to data. These models allow predictions about intervention effects on outcomes. Estimating counterfactuals, hypothetical outcomes if a specific treatment were assigned, is achievable through ML algorithms. By utilizing data from control groups, ML algorithms can estimate the outcome of a treatment for an individual [ 121 ] . Controlling for confounding variables, which can affect both the outcome and treatment, is achievable through ML algorithms. Including these variables as covariates in a regression model allows for confounding adjustment.
Personalized medicine benefits from ML algorithms that tailor treatments to individuals based on their unique characteristics and causal relationships between treatment and outcome [ 173 ] . Genetic profiles and medical histories can be employed to predict the treatment’s impact on a specific individual.
Causal discovery utilizes ML algorithms to unveil causal relationships between variables by analyzing data patterns [ 174 ] . For instance, genomics and clinical data can be used to identify causal links between genetic variants and diseases.
 Additional commonly used approach in ML is ensemble methods. Ensemble methods involve combining multiple models or predictions to improve overall performance and make more accurate predictions [ 175 ] . This can be done through techniques such as bagging, boosting, or stacking. Ensemble methods have been successful in various domains and can be applied to both classification and regression tasks. One of the key ideas for using Ensmeble in the context of causality is to reduce uncertainty in causal inferences by averaging plausible estimates from diverse models encoding alternative assumptions.
 Following by are several approaches for building causal Ensembles. Structural Causal Model (SCM) ensemble involves training multiple SCMs using different variable orderings and exclusion restrictions, and then taking a consensus from the ensemble [ 176 ] . Another approach is the Potential Outcomes ensemble, where individual treatment effect estimates are averaged from models that incorporate different sets of control variables [ 177 ] .
The Counterfactual Gaussian Process Ensemble utilizes Gaussian Process (GP) regression [ 178 , 179 ] models to establish relationships between treatments and outcomes, and the ensemble estimates the average causal effect by marginalizing over these models. Additional usage of ensemble that yields more accurate results can be found in decision causality trees and forests [ 180 ] as well as causal rules [ 181 ] .
In the Heterogeneous Effect Modeling Ensemble, separate treatment-stratified models are built to estimate effect modifiers, and the subgroup estimates are then combined [ 182 ] .
 Encoder-decoder is a common used architecture in DL. Encoder-decoder models have effectively been used for causality reasoning from natural language [ 183 ] . Cause-and-Effect Pair Mining (CEPM) [ 184 ] is an unsupervised model that learns relationships without labels by generating candidate pairs with an encoder-decoder and scoring them. Causal Inference over Natural Language (COIN) [ 185 ] also uses an encoder-decoder to produce implications and explanations, representing causality as queryable graphs. Some models not only propose claims but validate them via an encoder-decoder rationale generation component. Additionally, counterfactual text can be created with encoder-decoders by suggesting alterations to a potential cause or effect [ 186 ] . These techniques represent causality and help justify relationships, revealing how encoder-decoders are well-suited for understanding causality from language.
 ML algorithms enhance the interpretability of causal models by shedding light on the causal relationships between variables [ 187 ] . Identifying the most impactful causal relationships can be accomplished by analyzing the magnitude of their effects.
Time-series analysis benefits from ML algorithms to study causal relationships between variables that change over time [ 188 ] . Economic data, for example, can be employed to predict the effect of policy interventions using ML algorithms.
ML algorithms facilitate causal inference even with incomplete or missing data. By leveraging observed data patterns, missing data can be imputed using ML algorithms.
The above mentioned approaches and examples exemplify how ML and DL can be harnessed to explore causality within data. Integrating ML with causal inference allows for unveiling causal relationships between variables, making predictions regarding intervention effects, and applying these insights in various domains such as healthcare, finance, and economics.
In previous sections, we mentioned the gap between GA and causal inference approaches. While there are approaches that tries to connect between the two [ 189 , 190 , 191 ] .

 
 
 

## 13 Causality and Generative AI (GAI)

 
 Causality plays a crucial role in explaining past results, but it can also support generated data. The integration between causality and emerging GAI plays an important role in research and applications. 
 Various GAI approaches incorporate causal reasoning, such as Causal Bayesian Networks [ 192 ] , SCM [ 193 ] , Counterfactual Models, Causal Trees [ 194 ] , and CausalGANs [ 195 ] . Furthermore, there are ongoing research areas related to causality, including Causal Discovery, Interventional Models [ 196 ] , Counterfactual Reasoning, Transportability, Causal Disentanglement [ 197 ] , Longitudinal and Time Series Data [ 198 ] , and combining Causality and DL [ 199 ] .
There is still much work to be done in scaling up causal learning methods to handle large, complex real-world datasets while maintaining properties like robustness, modularity, and interpretability.
Several commonly used causal generative models include SCMs [ 193 ] , Causal Bayesian Networks (CBNs) [ 200 ] , Potential Outcome Models/Counterfactuals [ 201 ] , Markov Decision Processes (MDPs), Causal Additive Models (CAMs) [ 202 ] , Generative Adversarial Networks with Backdoor Adjustment (GAN-BA) [ 203 ] , Latent Causal Confounder Models [ 204 ] , Causal Inference Neural Networks (CINN) [ 205 ] , and Self-Supervised Causal Discovery (SSCD) [ 166 ] .
 These models formalize different causal assumptions to simulate interventions, estimate counterfactuals, and remove confounding bias to infer direct and indirect causal effects. Incorporating causal models into GAI can lead to advancements such as causal language models, structural causal Variational AutoEncoders (VAEs) [ 206 ] , causal infilling, Conditional Generative Adversarial Network (cGAN) with causal supervision [ 207 ] , contrastive generation, causal video prediction, and counterfactual generations. These examples demonstrate how explicit causal reasoning brings interpretability, interventional reasoning, and robustness benefits compared to traditional associative generative models.

 
 
 

## 14 Causality and Fuzzy Logic

 
 Causality and fuzzy logic are two interconnected concepts that have been extensively explored and integrated in various research studies and practical applications [ 208 ] . One such concept is fuzzy causality, which combines causality and fuzzy logic to reason about causal relationships in situations where the precise knowledge of such relationships is lacking [ 209 ] . Fuzzy causality provides a means to represent incomplete or uncertain information regarding the causal connections between variables, and it finds utility in decision-making processes and predictive modeling.
 Another area of investigation is causal fuzzy systems, which merge causal inference with fuzzy logic to analyze intricate systems characterized by uncertain or imprecise causal relationships [ 210 ] . By employing fuzzy sets and fuzzy logic, causal fuzzy systems enable the representation of the causal associations between variables. They find application in diverse domains such as control systems, robotics, and medical decision-making.
 Fuzzy causal models, on the other hand, are statistical models that utilize fuzzy logic to represent causal relationships between variables [ 211 , 212 ] . These models prove valuable when dealing with situations where the causal connections are uncertain or imprecise, allowing the inference of causal relationships from observational data.
The research field of fuzzy decision-making combines fuzzy logic and decision-making to investigate the process of decision-making in scenarios with incomplete or uncertain information [ 213 ] . Fuzzy decision-making techniques find applications in various domains, including medical, financial, and environmental decision-making, where the imprecision or uncertainty of information plays a significant role.
Lastly, fuzzy cognitive maps are cognitive models that employ fuzzy logic to represent the causal relationships between variables within a decision-making process [ 214 , 210 ] . These maps offer insights into decision-making under incomplete or uncertain information and aid in the design of decision-support systems capable of handling imprecise or uncertain data.

 
 
 

## 15 Conclusions and Future Directions

 
 AI and ML today rely primarily on statistical correlations in data to infer causation, whereas humans understand causation through intuitive theories about how the world works based on mechanisms, context, and common sense knowledge. Humans also view causation dynamically and consider uncertainties like hidden or confounding variables. Most current AI/ML approaches take a limited, static view of causation directly from observational data alone. We estimate one of the future challenges is to close this gap by endowing AI with capacities that mimic human causal reasoning abilities more closely. Giving systems mechanisms for broader domain and common sense knowledge, causal discovery algorithms accounting for correlation vs causation, representing and reasoning about uncertainties and unobserved variables, and continually refining beliefs based on new evidence could help align how causality is understood by AI systems with the more robust and flexible understanding of causality humans naturally possess. 
 In the realm of causal inference, various research areas can contribute to narrowing the gap between AI/ML causality and human causality. One such area is causal representation learning, which focuses on developing methods to learn embeddings or representations of variables that encode causal relationships and mechanisms. By capturing semantic similarity related to direct or indirect causality, these models can assist in tasks such as causal inference and structure learning. However, real-world data’s lack of perfect causal labels presents a challenge in this domain.
Another important research avenue is causal time series modeling, which extends existing causal graph models to incorporate temporal dependencies and feedback loops in time series data. This extension allows for the modeling of dynamic changes over time, enabling unified frameworks for tasks like time series forecasting, anomaly detection, and counterfactual analysis. Addressing non-stationarities within the data presents a significant challenge in this field.
Causal reinforcement learning is another promising area, involving the integration of causal reasoning into reinforcement learning. By incorporating causality, agents can better understand how their actions propagate consequences in the environment. This integration has the potential to enhance offline evaluation, generalization, and safe exploration. However, developing scalable causal models to handle complex and dynamic environments remains a challenge. 
 Additionally, causal discovery at scale is a critical area of research. The goal is to develop scalable algorithms capable of handling large-scale datasets with hundreds or thousands of variables. Tech niques such as two-stage learning approaches, variable clustering before structure search, and random projection or sampling can enable causal analysis of complex real-world systems. Thorough validation using synthetic and real-world benchmarks is essential in this context.
By pursuing these research directions, we can strive to bridge the gap between AI/ML causality and human causality, fostering a deeper understanding of causal relationships and paving the way for more reliable and interpretable AI systems.

 
 
 

## Acknowledgements

 
 We would like to thank Aleksander Molak for his feedback and remarks to the paper.

 
 
 

## References

 
 
 [1] 
 
J. Y. Halpern, “A modification of the halpern-pearl definition of causality,” arXiv preprint arXiv:1505.00162 , 2015.

 

 
 [2] 
 
J. Pearl, “The seven tools of causal inference, with reflections on machine learning,” Communications of the ACM , vol. 62, no. 3, pp. 54–60, 2019.

 

 
 [3] 
 
E. Bareinboim, J. D. Correa, D. Ibeling, and T. Icard, “On pearl’s hierarchy and the foundations of causal inference,” in Probabilistic and causal inference: the works of judea pearl , 2022, pp. 507–556.

 

 
 [4] 
 
D. Ibeling and T. Icard, “A topological perspective on causal inference,” CoRR , vol. abs/2107.08558, 2021. [Online]. Available: https://arxiv.org/abs/2107.08558

 

 
 [5] 
 
I. Shpitser and J. Pearl, “Complete identification methods for the causal hierarchy,” Journal of Machine Learning Research , vol. 9, pp. 1941–1979, 2008.

 

 
 [6] 
 
P. Illari and F. Russo, Causality: Philosophical theory meets scientific practice . OUP Oxford, 2014.

 

 
 [7] 
 
D. H. Jonassen and I. G. Ionas, “Designing effective supports for causal reasoning,” Educational Technology Research and Development , vol. 56, pp. 287–308, 2008.

 

 
 [8] 
 
S. S. Alhadad, “Visualizing data to support judgement, inference, and decision making in learning analytics: Insights from cognitive psychology and visualization science,” Journal of Learning Analytics , vol. 5, no. 2, pp. 60–85, 2018.

 

 
 [9] 
 
M. Friedman, Dynamics of reason . Csli Publications Stanford, 2001.

 

 
 [10] 
 
C. C. M. de Moura Belo, Chance and determinism in Avicenna and Averroes . Brill, 2007, vol. 69.

 

 
 [11] 
 
R. E. Rubenstein, Aristotle’s children: how Christians, Muslims, and Jews rediscovered ancient wisdom and illuminated the Middle Ages . HMH, 2004.

 

 
 [12] 
 
F. Bacon, Empiricism, Associationism, and Utilitarianism . Taylor and Francis Group, 2013.

 

 
 [13] 
 
S. A. Mulaik, “A brief history of the philosophical foundations of exploratory factor analysis,” Multivariate Behavioral Research , vol. 22, no. 3, pp. 267–305, 1987.

 

 
 [14] 
 
R. Descartes et al. , The Correspondence between Princess Elisabeth of Bohemia and René Descartes . University of Chicago Press, 2007.

 

 
 [15] 
 
G. Strawson, “David hume: objects and power,” in The New Hume Debate . Routledge, 2002, pp. 43–63.

 

 
 [16] 
 
L. Allais, “Kant’s one world: Interpreting’transcendental idealism’,” British Journal for the History of Philosophy , vol. 12, no. 4, pp. 655–684, 2004.

 

 
 [17] 
 
D. Caramani, Introduction to the comparative method with Boolean algebra . Sage publications, 2008.

 

 
 [18] 
 
G. H. Von Wright, Explanation and understanding . Cornell University Press, 2004.

 

 
 [19] 
 
T. J. McKeown, “Case studies and the statistical worldview: Review of king, keohane, and verba’s designing social inquiry: Scientific inference in qualitative research,” International organization , vol. 53, no. 1, pp. 161–190, 1999.

 

 
 [20] 
 
J. Pearl, “Embracing causality in formal reasoning,” in Proceedings of the sixth National conference on Artificial intelligence-Volume 1 , 1987, pp. 369–373.

 

 
 [21] 
 
P. Spirtes, C. N. Glymour, and R. Scheines, Causation, prediction, and search . MIT press, 2000.

 

 
 [22] 
 
C. Glymour, R. Scheines, P. Spirtes, and K. Kelly, “Discovering causal structure: Artificial intelligence,” Philosophy of science, and Statistical Modeling , p. 394, 1987.

 

 
 [23] 
 
V. Didelez, “Perspective on interviews with heckman, pearl, robins and rubin,” Observational Studies , vol. 8, no. 2, pp. 95–104, 2022.

 

 
 [24] 
 
J. Sekhon, “The neyman—rubin model of causal inference and estimation via matching methods,” The Oxford handbook of political methodology , 2008.

 

 
 [25] 
 
P. Spirtes and K. Zhang, “Causal discovery and inference: concepts and recent methodological advances,” in Applied informatics , vol. 3, no. 1. SpringerOpen, 2016, pp. 1–28.

 

 
 [26] 
 
A. Gelman, “Causality and statistical learning,” 2011.

 

 
 [27] 
 
J. M. Mooij, J. Peters, D. Janzing, J. Zscheischler, and B. Schölkopf, “Distinguishing cause from effect using observational data: methods and benchmarks,” The Journal of Machine Learning Research , vol. 17, no. 1, pp. 1103–1204, 2016.

 

 
 [28] 
 
S. Athey and G. W. Imbens, “Machine learning methods for estimating heterogeneous causal effects,” stat , vol. 1050, no. 5, pp. 1–26, 2015.

 

 
 [29] 
 
D. Card, “The causal effect of education on earnings,” Handbook of labor economics , vol. 3, pp. 1801–1863, 1999.

 

 
 [30] 
 
J. D. Angrist, G. W. Imbens, and D. B. Rubin, “Identification of causal effects using instrumental variables,” Journal of the American statistical Association , vol. 91, no. 434, pp. 444–455, 1996.

 

 
 [31] 
 
J. Reiss, “Suppes’ probabilistic theory of causality and causal inference in economics,” in Patrick Suppes, Economics, and Economic Methodology . Routledge, 2018, pp. 53–68.

 

 
 [32] 
 
P. Suppes, “The measurement of belief,” Journal of the Royal Statistical Society: Series B (Methodological) , vol. 36, no. 2, pp. 160–175, 1974.

 

 
 [33] 
 
V. K. Raghu, J. D. Ramsey, A. Morris, D. V. Manatakis, P. Sprites, P. K. Chrysanthis, C. Glymour, and P. V. Benos, “Comparison of strategies for scalable causal discovery of latent variable models from mixed data,” International journal of data science and analytics , vol. 6, pp. 33–45, 2018.

 

 
 [34] 
 
H. E. Brady, “Models of causal inference: Going beyond the neyman-rubin-holland theory,” in Annual Meetings of the Political Methodology Group , 2002.

 

 
 [35] 
 
J. Pearl, “Causal diagrams for empirical research,” Biometrika , vol. 82, no. 4, pp. 669–688, 1995.

 

 
 [36] 
 
——, “Trygve haavelmo and the emergence of causal calculus,” Econometric Theory , vol. 31, no. 1, pp. 152–179, 2015.

 

 
 [37] 
 
J. Cheng and R. Greiner, “Comparing bayesian network classifiers,” arXiv preprint arXiv:1301.6684 , 2013.

 

 
 [38] 
 
Y. Peng and J. A. Reggia, “A probabilistic causal model for diagnostic problem solving part i: Integrating symbolic causal inference with numeric probabilistic inference,” IEEE Transactions on Systems, Man, and Cybernetics , vol. 17, no. 2, pp. 146–162, 1987.

 

 
 [39] 
 
D. L. Weed, “Interpreting epidemiological evidence: how meta-analysis and causal inference methods are related,” International Journal of Epidemiology , vol. 29, no. 3, pp. 387–390, 2000.

 

 
 [40] 
 
D. Galles and J. Pearl, “An axiomatic characterization of causal counterfactuals,” Foundations of Science , vol. 3, pp. 151–182, 1998.

 

 
 [41] 
 
J. Vennekens, M. Bruynooghe, and M. Denecker, “Embracing events in causal modelling: Interventions and counterfactuals in cp-logic,” in European workshop on logics in artificial intelligence . Springer, 2010, pp. 313–325.

 

 
 [42] 
 
J. M. Robins, A. Rotnitzky, and D. O. Scharfstein, “Sensitivity analysis for selection bias and unmeasured confounding in missing data and causal inference models,” in Statistical models in epidemiology, the environment, and clinical trials . Springer, 2000, pp. 1–94.

 

 
 [43] 
 
M. L. Petersen, S. E. Sinisi, and M. J. van der Laan, “Estimation of direct causal effects,” Epidemiology , vol. 17, no. 3, pp. 276–284, 2006.

 

 
 [44] 
 
T. Heskes, E. Sijben, I. G. Bucur, and T. Claassen, “Causal shapley values: Exploiting causal knowledge to explain individual predictions of complex models,” Advances in neural information processing systems , vol. 33, pp. 4778–4789, 2020.

 

 
 [45] 
 
P. Schwab and W. Karlen, “Cxplain: Causal explanations for model interpretation under uncertainty,” Advances in neural information processing systems , vol. 32, 2019.

 

 
 [46] 
 
J. Schaffer, “Anchoring as grounding: On epstein’s the ant trap,” Philosophy and Phenomenological Research , vol. 99, no. 3, pp. 749–767, 2019.

 

 
 [47] 
 
G. Davey Smith and G. Hemani, “Mendelian randomization: genetic anchors for causal inference in epidemiological studies,” Human molecular genetics , vol. 23, no. R1, pp. R89–R98, 2014.

 

 
 [48] 
 
G. Gurevich, D. Kliger, and B. Weiner, “The role of attribution of causality in economic decision making,” The Journal of Socio-Economics , vol. 41, no. 4, pp. 439–444, 2012.

 

 
 [49] 
 
X. Huang and J. Marques-Silva, “The inadequacy of shapley values for explainability,” 2023.

 

 
 [50] 
 
R. Foraita, J. Friemel, K. Günther, T. Behrens, J. Bullerdiek, R. Nimzyk, W. Ahrens, and V. Didelez, “Causal discovery of gene regulation with incomplete data,” Journal of the Royal Statistical Society Series A: Statistics in Society , vol. 183, no. 4, pp. 1747–1775, 2020.

 

 
 [51] 
 
J. Kelly, C. Berzuini, B. Keavney, M. Tomaszewski, and H. Guo, “A review of causal discovery methods for molecular network analysis,” Molecular Genetics and Genomic Medicine , vol. 10, no. 10, Oct. 2022.

 

 
 [52] 
 
R. O. Ness, K. Sachs, P. Mallick, and O. Vitek, “A bayesian active learning experimental design for inferring signaling networks,” in Research in Computational Molecular Biology: 21st Annual International Conference, RECOMB 2017, Hong Kong, China, May 3-7, 2017, Proceedings 21 . Springer, 2017, pp. 134–156.

 

 
 [53] 
 
C. Glymour, K. Zhang, and P. Spirtes, “Review of causal discovery methods based on graphical models,” Frontiers in genetics , vol. 10, p. 524, 2019.

 

 
 [54] 
 
C. Gao, Y. Zheng, W. Wang, F. Feng, X. He, and Y. Li, “Causal inference in recommender systems: A survey and future directions,” arXiv preprint arXiv:2208.12397 , 2022.

 

 
 [55] 
 
D. Goldenberg, J. Albert, L. Bernardi, and P. Estevez, “Free lunch! retrospective uplift modeling for dynamic promotions recommendation within roi constraints,” in Fourteenth ACM Conference on Recommender Systems , ser. RecSys ’20. ACM, Sep. 2020. [Online]. Available: http://dx.doi.org/10.1145/3383313.3412215

 

 
 [56] 
 
F. Moraes, H. M. Proença, A. Kornilova, J. Albert, and D. Goldenberg, “Uplift modeling: from causal inference to personalization,” 2023.

 

 
 [57] 
 
D. Xu, Y. Wu, S. Yuan, L. Zhang, and X. Wu, “Achieving causal fairness through generative adversarial networks,” in Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence , 2019.

 

 
 [58] 
 
L. Gultchin, “Casual and trustworthy machine learning: methods and applications,” Ph.D. dissertation, University of Oxford, 2023.

 

 
 [59] 
 
D. Plecko and E. Bareinboim, “Causal fairness analysis,” arXiv preprint arXiv:2207.11385 , 2022.

 

 
 [60] 
 
E. Bareinboim, A. Forney, and J. Pearl, “Bandits with unobserved confounders: a causal approach,” in Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 1 , ser. NIPS’15. Cambridge, MA, USA: MIT Press, 2015, p. 1342–1350.

 

 
 [61] 
 
S. Lee and E. Bareinboim, “Structural causal bandits: Where to intervene?” Advances in neural information processing systems , vol. 31, 2018.

 

 
 [62] 
 
A. K. Lampinen, N. Roy, I. Dasgupta, S. C. Chan, A. Tam, J. Mcclelland, C. Yan, A. Santoro, N. C. Rabinowitz, J. Wang et al. , “Tell me why! explanations support learning relational and causal structure,” in International Conference on Machine Learning . PMLR, 2022, pp. 11 868–11 890.

 

 
 [63] 
 
F. Lattimore, T. Lattimore, and M. D. Reid, “Causal bandits: Learning good interventions via causal inference,” 2016.

 

 
 [64] 
 
J. Richens and T. Everitt, “Robust agents learn causal world models,” 2024.

 

 
 [65] 
 
A. Boustati, H. Chockler, and D. C. McNamee, “Transfer learning with causal counterfactual reasoning in decision transformers,” arXiv preprint arXiv:2110.14355 , 2021.

 

 
 [66] 
 
M. Edmonds, X. Ma, S. Qi, Y. Zhu, H. Lu, and S.-C. Zhu, “Theory-based causal transfer: Integrating instance-level induction and abstract-level structure learning,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 34, no. 02, 2020, pp. 1283–1291.

 

 
 [67] 
 
A. N. Carey and X. Wu, “The causal fairness field guide: Perspectives from social and formal sciences,” Frontiers in Big Data , vol. 5, p. 892837, 2022.

 

 
 [68] 
 
Z. Chen, M. Xu, B. Gao, G. Sugihara, F. Shen, Y. Cai, A. Li, Q. Wu, L. Yang, Q. Yao et al. , “Causation inference in complicated atmospheric environment,” Environmental Pollution , vol. 303, p. 119057, 2022.

 

 
 [69] 
 
J. Pearl, “Causal inference in statistics: An overview,” Statistics Surveys , 2009.

 

 
 [70] 
 
J. Zhang, “Causal inference and reasoning in causally insufficient systems,” Ph.D. dissertation, Citeseer, 2006.

 

 
 [71] 
 
P. Hosseini, D. A. Broniatowski, and M. Diab, “Predicting directionality in causal relations in text,” arXiv preprint arXiv:2103.13606 , 2021.

 

 
 [72] 
 
P. Nadathur and S. Lauer, “Causal necessity, causal sufficiency, and the implications of causative verbs,” Glossa: a journal of general linguistics , vol. 5, no. 1, 2020.

 

 
 [73] 
 
T. Harinen, “Mutual manipulability and causal inbetweenness,” Synthese , vol. 195, pp. 35–54, 2018.

 

 
 [74] 
 
J. Woodward, “Causation and manipulability,” The Stanford Encyclopedia of Philosophy , 2016.

 

 
 [75] 
 
P. A. White, “The causal asymmetry.” Psychological review , vol. 113, no. 1, p. 132, 2006.

 

 
 [76] 
 
E. Eells and E. Sober, “Probabilistic causality and the question of transitivity,” Philosophy of science , vol. 50, no. 1, pp. 35–57, 1983.

 

 
 [77] 
 
I. Bica, D. Jarrett, and M. van der Schaar, “Invariant causal imitation learning for generalizable policies,” Advances in Neural Information Processing Systems , vol. 34, pp. 3952–3964, 2021.

 

 
 [78] 
 
J. W. Irwin, “The effects of explicitness and clause order on the comprehension of reversible causal relationships,” Reading Research Quarterly , pp. 477–488, 1980.

 

 
 [79] 
 
L. Bertossi, J. Li, M. Schleich, D. Suciu, and Z. Vagena, “Causality-based explanation of classification outcomes,” in Proceedings of the Fourth International Workshop on Data Management for End-to-End Machine Learning , 2020, pp. 1–10.

 

 
 [80] 
 
D. A. Lagnado, T. Gerstenberg, and R. Zultan, “Causal responsibility and counterfactuals,” Cognitive science , vol. 37, no. 6, pp. 1036–1073, 2013.

 

 
 [81] 
 
J. D. Schenker and P. D. Rumrill Jr, “Causal-comparative research designs,” Journal of vocational rehabilitation , vol. 21, no. 3, pp. 117–121, 2004.

 

 
 [82] 
 
N. Ganguly, D. Fazlija, M. Badar, M. Fisichella, S. Sikdar, J. Schrader, J. Wallat, K. Rudra, M. Koubarakis, G. K. Patro et al. , “A review of the role of causality in developing trustworthy ai systems,” arXiv preprint arXiv:2302.06975 , 2023.

 

 
 [83] 
 
B. Befani, “Models of causality and causal inference,” Broadening the Range of Designs and Methods for Impact Evaluation , vol. 38, 2012.

 

 
 [84] 
 
N. Cartwright, “Modularity: It can-and generally does-fail,” Stochastic Causality , 2001.

 

 
 [85] 
 
D. J. Greiner and D. B. Rubin, “Causal effects of perceived immutable characteristics,” Review of Economics and Statistics , vol. 93, no. 3, pp. 775–785, 2011.

 

 
 [86] 
 
M. H. Olesen, D. K. Thomsen, A. Schnieber, and J. Tønnesvang, “Distinguishing general causality orientations from personality traits,” Personality and individual differences , vol. 48, no. 5, pp. 538–543, 2010.

 

 
 [87] 
 
C. Ksir and C. L. Hart, “Correlation still does not imply causation,” The Lancet Psychiatry , vol. 3, no. 5, p. 401, 2016.

 

 
 [88] 
 
I. Guyon et al. , “Practical feature selection: from correlation to causality,” Mining massive data sets for security: advances in data mining, search, social networks and text mining, and their applications to security , pp. 27–43, 2008.

 

 
 [89] 
 
J. Pearl and D. Mackenzie, The Book of Why . New York: Basic Books, 2018.

 

 
 [90] 
 
J. Y. Halpern, Actual Causality . Cambridge, MA: MIT Press, 2016.

 

 
 [91] 
 
E. Bareinboim and J. Pearl, “A general algorithm for deciding transportability of experimental results,” Journal of Causal Inference , vol. 1, no. 1, p. 107–134, May 2013. [Online]. Available: http://dx.doi.org/10.1515/jci-2012-0004

 

 
 [92] 
 
J. Peters, D. Janzing, and B. Schölkopf, Elements of Causal Inference: Foundations and Learning Algorithms , ser. Adaptive Computation and Machine Learning. Cambridge, MA: MIT Press, 2017. [Online]. Available: https://mitpress.mit.edu/books/elements-causal-inference

 

 
 [93] 
 
L. Yao, Z. Chu, S. Li, Y. Li, J. Gao, and A. Zhang, “A survey on causal inference,” ACM Transactions on Knowledge Discovery from Data (TKDD) , vol. 15, no. 5, pp. 1–46, 2021.

 

 
 [94] 
 
J. Pearl, “[bayesian analysis in expert systems]: comment: graphical models, causality and intervention,” Statistical Science , vol. 8, no. 3, pp. 266–269, 1993.

 

 
 [95] 
 
A. Ghassami, A. Yang, I. Shpitser, and E. T. Tchetgen, “Causal inference with hidden mediators,” arXiv preprint arXiv:2111.02927 , 2021.

 

 
 [96] 
 
D. B. Rubin, “Causal inference using potential outcomes: Design, modeling, decisions,” Journal of the American Statistical Association , vol. 100, no. 469, pp. 322–331, 2005.

 

 
 [97] 
 
S. O’Neill, N. Kreif, R. Grieve, M. Sutton, and J. S. Sekhon, “Estimating causal effects: considering three alternatives to difference-in-differences estimation,” Health Services and Outcomes Research Methodology , vol. 16, pp. 1–21, 2016.

 

 
 [98] 
 
Y. Fan, J. Chen, G. Shirkey, R. John, S. R. Wu, H. Park, and C. Shao, “Applications of structural equation modeling (sem) in ecological studies: an updated review,” Ecological Processes , vol. 5, pp. 1–12, 2016.

 

 
 [99] 
 
J. Pearl, “Graphs, causality, and structural equation models,” Sociological Methods Research , vol. 27, no. 2, pp. 226–284, 1998.

 

 
 [100] 
 
M. Caliendo and S. Kopeinig, “Some practical guidance for the implementation of propensity score matching,” Journal of economic surveys , vol. 22, no. 1, pp. 31–72, 2008.

 

 
 [101] 
 
M. Li, “Using the propensity score method to estimate causal effects: A review and practical guide,” Organizational Research Methods , vol. 16, no. 2, pp. 188–226, 2013.

 

 
 [102] 
 
C. Trampusch and B. Palier, “Between x and y: how process tracing contributes to opening the black box of causality,” New political economy , vol. 21, no. 5, pp. 437–454, 2016.

 

 
 [103] 
 
L. Hood, L. Bloom, and C. J. Brainerd, “What, when, and how about why: A longitudinal study of early expressions of causality,” Monographs of the society for research in child development , pp. 1–47, 1979.

 

 
 [104] 
 
P. Duan, F. Yang, T. Chen, and S. L. Shah, “Detection of direct causality based on process data,” in 2012 American Control Conference (ACC) . IEEE, 2012, pp. 3522–3527.

 

 
 [105] 
 
J.-M. Dufour and A. Taamouti, “Short and long run causality measures: Theory and inference,” Journal of Econometrics , vol. 154, no. 1, pp. 42–58, 2010.

 

 
 [106] 
 
V. A. Vakorin, O. A. Krakovska, and A. R. McIntosh, “Confounding effects of indirect connections on causality estimation,” Journal of neuroscience methods , vol. 184, no. 1, pp. 152–160, 2009.

 

 
 [107] 
 
D. Koutsoyiannis, C. Onof, A. Christofides, and Z. W. Kundzewicz, “Revisiting causality using stochastics: 1. theory,” Proceedings of The Royal Society A , vol. 478, no. 2261, p. 20210835, 2022.

 

 
 [108] 
 
J. Dul, “Necessary condition analysis (nca) logic and methodology of “necessary but not sufficient” causality,” Organizational Research Methods , vol. 19, no. 1, pp. 10–52, 2016.

 

 
 [109] 
 
P. Petraitis, A. Dunham, and P. Niewiarowski, “Inferring multiple causality: the limitations of path analysis,” Functional ecology , pp. 421–431, 1996.

 

 
 [110] 
 
J. Pearl, “Structural and probabilistic causality,” in Psychology of learning and motivation . Elsevier, 1996, vol. 34, pp. 393–435.

 

 
 [111] 
 
P. K. Tyagi and T. R. Wotruba, “An exploratory study of reverse causality relationships among sales force turnover variables,” Journal of the Academy of Marketing Science , vol. 21, pp. 143–153, 1993.

 

 
 [112] 
 
F. C. Keil, “Explanation and understanding,” Annu. Rev. Psychol. , vol. 57, pp. 227–254, 2006.

 

 
 [113] 
 
S. Häggqvist, “Kinds, projectibility and explanation,” Croatian journal of philosophy , no. 13, pp. 71–87, 2005.

 

 
 [114] 
 
N. Weinberger, “Intervening and letting go: On the adequacy of equilibrium causal models,” 2021. [Online]. Available: https://philsci-archive.pitt.edu/19558/

 

 
 [115] 
 
W. D. Gunter and K. Daly, “Causal or spurious: Using propensity score matching to detangle the relationship between violent video games and violent behavior,” Computers in Human Behavior , vol. 28, no. 4, pp. 1348–1355, 2012.

 

 
 [116] 
 
K. Imai, L. Keele, and D. Tingley, “A general approach to causal mediation analysis.” Psychological methods , vol. 15, no. 4, p. 309, 2010.

 

 
 [117] 
 
A. D. Wu and B. D. Zumbo, “Understanding and using mediators and moderators,” Social Indicators Research , vol. 87, pp. 367–392, 2008.

 

 
 [118] 
 
H. M. Blalock Jr, Causal inferences in nonexperimental research . UNC Press Books, 2018.

 

 
 [119] 
 
H. Álvarez-Martínez and E. Pérez-Campos, “Causality in medicine,” Gaceta médica de México , vol. 140, no. 4, pp. 467–472, 2004.

 

 
 [120] 
 
M. A. Hernán and T. J. VanderWeele, “Compound treatments and transportability of causal inference,” Epidemiology (Cambridge, Mass.) , vol. 22, no. 3, p. 368, 2011.

 

 
 [121] 
 
M. Prosperi, Y. Guo, M. Sperrin, J. S. Koopman, J. S. Min, X. He, S. Rich, M. Wang, I. E. Buchan, and J. Bian, “Causal inference and counterfactual prediction in machine learning for actionable healthcare,” Nature Machine Intelligence , vol. 2, no. 7, pp. 369–375, 2020.

 

 
 [122] 
 
M. R. Kosorok and E. B. Laber, “Precision medicine,” Annual review of statistics and its application , vol. 6, pp. 263–286, 2019.

 

 
 [123] 
 
K. Hlaváčková-Schindler, M. Paluš, M. Vejmelka, and J. Bhattacharya, “Causality detection based on information-theoretic approaches in time series analysis,” Physics Reports , vol. 441, no. 1, pp. 1–46, 2007.

 

 
 [124] 
 
V. A. Atanasov and B. S. Black, “Shock-based causal inference in corporate finance and accounting research,” Critical Finance Review , vol. 5, pp. 207–304, 2016.

 

 
 [125] 
 
R. Moraffah, P. Sheth, M. Karami, A. Bhattacharya, Q. Wang, A. Tahir, A. Raglin, and H. Liu, “Causal inference for time series analysis: Problems, methods and evaluation,” Knowledge and Information Systems , vol. 63, pp. 3041–3085, 2021.

 

 
 [126] 
 
I. D. Gow, D. F. Larcker, and P. C. Reiss, “Causal inference in accounting research,” Journal of Accounting Research , vol. 54, no. 2, pp. 477–523, 2016.

 

 
 [127] 
 
M. J. Kusner, J. Loftus, C. Russell, and R. Silva, “Counterfactual fairness,” Advances in neural information processing systems , vol. 30, 2017.

 

 
 [128] 
 
R. J. Murnane and J. B. Willett, Methods matter: Improving causal inference in educational and social science research . Oxford University Press, 2010.

 

 
 [129] 
 
T. Zajonc, “Essays on causal inference for public policy,” Ph.D. dissertation, Harvard University, 2012.

 

 
 [130] 
 
Y. Wang, D. Liang, L. Charlin, and D. M. Blei, “Causal inference for recommender systems,” in Proceedings of the 14th ACM Conference on Recommender Systems , 2020, pp. 426–431.

 

 
 [131] 
 
Y. Vivek, V. Ravi, A. A. Mane, and L. R. Naidu, “Explainable artificial intelligence and causal inference based atm fraud detection,” arXiv preprint arXiv:2211.10595 , 2022.

 

 
 [132] 
 
S. C. Smith and S. Ramamoorthy, “Counterfactual explanation and causal inference in service of robustness in robot control,” in 2020 Joint IEEE 10th International Conference on Development and Learning and Epigenetic Robotics (ICDL-EpiRob) . IEEE, 2020, pp. 1–8.

 

 
 [133] 
 
J. Xu, K. Yin, J. M. Gregory, and L. Liu, “Causal inference for de-biasing motion estimation from robotic observational data,” in 2023 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2023, pp. 3008–3014.

 

 
 [134] 
 
M. Gangl, “Causal inference in sociological research,” Annual review of sociology , vol. 36, pp. 21–47, 2010.

 

 
 [135] 
 
H. R. Varian, “Causal inference in economics and marketing,” Proceedings of the National Academy of Sciences , vol. 113, no. 27, pp. 7310–7315, 2016.

 

 
 [136] 
 
S. Abel, Y. Tang, J. Singh, and E. Paek, “Applications of causal modeling in cybersecurity: An exploratory approach,” Advances in Science, Technology and Engineering Systems Journal , vol. 5, no. 3, pp. 380–387, 2020.

 

 
 [137] 
 
K. Kuang, L. Li, Z. Geng, L. Xu, K. Zhang, B. Liao, H. Huang, P. Ding, W. Miao, and Z. Jiang, “Causal inference,” Engineering , vol. 6, no. 3, pp. 253–263, 2020.

 

 
 [138] 
 
M. Rohmatillah and J.-T. Chien, “Causal confusion reduction for robust multi-domain dialogue policy.” in Interspeech , 2021, pp. 3221–3225.

 

 
 [139] 
 
P. Machamer, “Activities and causation: The metaphysics and epistemology of mechanisms,” International studies in the philosophy of science , vol. 18, no. 1, pp. 27–39, 2004.

 

 
 [140] 
 
G. Young, K. Nicholson, A. W. Kane, G. Young, and A. W. Kane, “Causality in psychology and law,” Causality of psychological injury: Presenting evidence in court , pp. 13–47, 2007.

 

 
 [141] 
 
M. M. Marini and B. Singer, “Causality in the social sciences,” Sociological methodology , vol. 18, pp. 347–409, 1988.

 

 
 [142] 
 
J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, and D. Sejdinovic, “Detecting and quantifying causal associations in large nonlinear time series datasets,” Science advances , vol. 5, no. 11, p. eaau4996, 2019.

 

 
 [143] 
 
C. S. Khoo, S. Chan, and Y. Niu, “Extracting causal knowledge from a medical database using graphical patterns,” in Proceedings of the 38th annual meeting of the association for computational linguistics , 2000, pp. 336–343.

 

 
 [144] 
 
A. Subbaswamy and S. Saria, “Counterfactual normalization: Proactively addressing dataset shift using causal mechanisms.” in UAI , 2018, pp. 947–957.

 

 
 [145] 
 
Z. Zhang, J. Zhou, W. Cao, and J. Zhang, “Causal inference with a quantitative exposure,” Statistical methods in medical research , vol. 25, no. 1, pp. 315–335, 2016.

 

 
 [146] 
 
G. A. Fox, “Practical causal inference for ecoepidemiologists,” Journal of Toxicology and Environmental Health, Part A Current Issues , vol. 33, no. 4, pp. 359–373, 1991.

 

 
 [147] 
 
G. King, C. Lucas, and R. A. Nielsen, “The balance-sample size frontier in matching methods for causal inference,” American journal of political science , vol. 61, no. 2, pp. 473–489, 2017.

 

 
 [148] 
 
K. M. Sheffield, N. A. Dreyer, J. F. Murray, D. E. Faries, and M. N. Klopchin, “Replication of randomized clinical trial results using real-world data: paving the way for effectiveness decisions,” Journal of Comparative Effectiveness Research , vol. 9, no. 15, pp. 1043–1050, 2020.

 

 
 [149] 
 
J. M. Robins, R. Scheines, P. Spirtes, and L. Wasserman, “Uniform consistency in causal inference,” Biometrika , vol. 90, no. 3, pp. 491–515, 2003.

 

 
 [150] 
 
R. S. Zimmermann, J. Borowski, R. Geirhos, M. Bethge, T. Wallis, and W. Brendel, “How well do feature visualizations support causal understanding of cnn activations?” Advances in Neural Information Processing Systems , vol. 34, pp. 11 730–11 744, 2021.

 

 
 [151] 
 
A. L. Duckworth, E. Tsukayama, and H. May, “Establishing causality using longitudinal hierarchical linear modeling: An illustration predicting achievement from self-control,” Social psychological and personality science , vol. 1, no. 4, pp. 311–317, 2010.

 

 
 [152] 
 
T. D. Cook, D. T. Campbell, and W. Shadish, Experimental and quasi-experimental designs for generalized causal inference . Houghton Mifflin Boston, MA, 2002, vol. 1195.

 

 
 [153] 
 
H. Liu, M. Chaudhary, and H. Wang, “Towards trustworthy and aligned machine learning: A data-centric survey with causality perspectives,” arXiv preprint arXiv:2307.16851 , 2023.

 

 
 [154] 
 
A. B. Arrieta, N. Díaz-Rodríguez, J. Del Ser, A. Bennetot, S. Tabik, A. Barbado, S. García, S. Gil-López, D. Molina, R. Benjamins et al. , “Explainable artificial intelligence (xai): Concepts, taxonomies, opportunities and challenges toward responsible ai,” Information fusion , vol. 58, pp. 82–115, 2020.

 

 
 [155] 
 
D. Janzing, L. Minorics, and P. Blöbaum, “Feature relevance quantification in explainable ai: A causal problem,” in International Conference on artificial intelligence and statistics . PMLR, 2020, pp. 2907–2916.

 

 
 [156] 
 
M. Naser, “An engineer’s guide to explainable artificial intelligence and interpretable machine learning: Navigating causality, forced goodness, and the false perception of inference,” Automation in Construction , vol. 129, p. 103821, 2021.

 

 
 [157] 
 
A. Holzinger, “Explainable ai and multi-modal causability in medicine,” i-com , vol. 19, no. 3, pp. 171–179, 2021.

 

 
 [158] 
 
Y.-L. Chou, C. Moreira, P. Bruza, C. Ouyang, and J. Jorge, “Counterfactuals and causability in explainable artificial intelligence: Theory, algorithms, and applications,” Information Fusion , vol. 81, pp. 59–83, 2022.

 

 
 [159] 
 
P. Madumal, T. Miller, L. Sonenberg, and F. Vetere, “Explainable reinforcement learning through a causal lens,” in Proceedings of the AAAI conference on artificial intelligence , vol. 34, no. 03, 2020, pp. 2493–2500.

 

 
 [160] 
 
J. M. Bishop, “Artificial intelligence is stupid and causal reasoning will not fix it,” Frontiers in Psychology , vol. 11, p. 2603, 2021.

 

 
 [161] 
 
T. J. Stewart, R. Janssen, and M. Van Herwijnen, “A genetic algorithm approach to multiobjective land use planning,” Computers Operations Research , vol. 31, no. 14, pp. 2293–2313, 2004.

 

 
 [162] 
 
Y. Demchenko, C. De Laat, and P. Membrey, “Defining architecture components of the big data ecosystem,” in 2014 International conference on collaboration technologies and systems (CTS) . IEEE, 2014, pp. 104–112.

 

 
 [163] 
 
R. Guo, L. Cheng, J. Li, P. R. Hahn, and H. Liu, “A survey of learning causality with data: Problems and methods,” ACM Computing Surveys (CSUR) , vol. 53, no. 4, pp. 1–37, 2020.

 

 
 [164] 
 
F. Mannering, C. R. Bhat, V. Shankar, and M. Abdel-Aty, “Big data, traditional data and the tradeoffs between prediction and causality in highway-safety analysis,” Analytic methods in accident research , vol. 25, p. 100113, 2020.

 

 
 [165] 
 
K. Arulkumaran, M. P. Deisenroth, M. Brundage, and A. A. Bharath, “Deep reinforcement learning: A brief survey,” IEEE Signal Processing Magazine , vol. 34, no. 6, pp. 26–38, 2017.

 

 
 [166] 
 
S. A. Sontakke, A. Mehrjou, L. Itti, and B. Schölkopf, “Causal curiosity: Rl agents discovering self-supervised experiments for causal representation learning,” in International conference on machine learning . PMLR, 2021, pp. 9848–9858.

 

 
 [167] 
 
S. J. Grimbly, J. Shock, and A. Pretorius, “Causal multi-agent reinforcement learning: Review and open problems,” arXiv preprint arXiv:2111.06721 , 2021.

 

 
 [168] 
 
Y. Lu, A. Meisami, and A. Tewari, “Efficient reinforcement learning with prior causal knowledge,” in Conference on Causal Learning and Reasoning . PMLR, 2022, pp. 526–541.

 

 
 [169] 
 
T. He, J. Gajcin, and I. Dusparic, “Causal counterfactuals for improving the robustness of reinforcement learning,” arXiv preprint arXiv:2211.05551 , 2022.

 

 
 [170] 
 
B. Schölkopf, “Causality for machine learning,” in Probabilistic and Causal Inference: The Works of Judea Pearl , 2022, pp. 765–804.

 

 
 [171] 
 
V. Ramachandra, “Deep learning for causal inference,” arXiv preprint arXiv:1803.00149 , 2018.

 

 
 [172] 
 
D. Kalainathan, O. Goudet, and R. Dutta, “Causal discovery toolbox: Uncovering causal relationships in python,” The Journal of Machine Learning Research , vol. 21, no. 1, pp. 1406–1410, 2020.

 

 
 [173] 
 
P. Sanchez, J. P. Voisey, T. Xia, H. I. Watson, A. Q. O’Neil, and S. A. Tsaftaris, “Causal machine learning for healthcare and precision medicine,” Royal Society Open Science , vol. 9, no. 8, p. 220638, 2022.

 

 
 [174] 
 
P. Arora, D. Boyne, J. J. Slater, A. Gupta, D. R. Brenner, and M. J. Druzdzel, “Bayesian networks for risk prediction using real-world data: a tool for precision medicine,” Value in Health , vol. 22, no. 4, pp. 439–445, 2019.

 

 
 [175] 
 
X. Dong, Z. Yu, W. Cao, Y. Shi, and Q. Ma, “A survey on ensemble learning,” Frontiers of Computer Science , vol. 14, pp. 241–258, 2020.

 

 
 [176] 
 
M. Li, R. Zhang, and K. Liu, “A new ensemble learning algorithm combined with causal analysis for bayesian network structural learning,” Symmetry , vol. 12, no. 12, p. 2054, 2020.

 

 
 [177] 
 
P. C. Austin, “Using ensemble-based methods for directly estimating causal effects: an investigation of tree-based g-computation,” Multivariate behavioral research , vol. 47, no. 1, pp. 115–135, 2012.

 

 
 [178] 
 
J. Z. Liu, J. Paisley, M.-A. Kioumourtzoglou, and B. A. Coull, “Adaptive ensemble learning of spatiotemporal processes with calibrated predictive uncertainty: A bayesian nonparametric approach,” arXiv preprint arXiv:1904.00521 , 2019.

 

 
 [179] 
 
A. Mishler and E. Kennedy, “Fade: Fair double ensemble learning for observable and counterfactual outcomes,” arXiv preprint arXiv:2109.00173 , 2021.

 

 
 [180] 
 
N. Younas, A. Ali, H. Hina, M. Hamraz, Z. Khan, and S. Aldahmani, “Optimal causal decision trees ensemble for improved prediction and causal inference,” IEEE Access , vol. 10, pp. 13 000–13 011, 2022.

 

 
 [181] 
 
K. Lee, F. J. Bargagli-Stoffi, and F. Dominici, “Causal rule ensemble: Interpretable inference of heterogeneous treatment effects,” arXiv preprint arXiv:2009.09036 , 2020.

 

 
 [182] 
 
F. J. Bargagli-Stoffi, R. Cadei, K. Lee, and F. Dominici, “Causal rule ensemble: Interpretable discovery and inference of heterogeneous causal effects,” arXiv preprint arXiv:2009.09036 , 2020.

 

 
 [183] 
 
T. Nayak, S. Sharma, Y. Butala, K. Dasgupta, P. Goyal, and N. Ganguly, “A generative approach for financial causality extraction,” in Companion Proceedings of the Web Conference 2022 , 2022, pp. 576–578.

 

 
 [184] 
 
O. Hassanzadeh, D. Bhattacharjya, M. Feblowitz, K. Srinivas, M. Perrone, S. Sohrabi, and M. Katz, “Answering binary causal questions through large-scale text mining: An evaluation using cause-effect pairs from human experts.” in IJCAI , 2019, pp. 5003–5009.

 

 
 [185] 
 
A. Feder, K. A. Keith, E. Manzoor, R. Pryzant, D. Sridhar, Z. Wood-Doughty, J. Eisenstein, J. Grimmer, R. Reichart, M. E. Roberts et al. , “Causal inference in natural language processing: Estimation, prediction, interpretation and beyond,” Transactions of the Association for Computational Linguistics , vol. 10, pp. 1138–1158, 2022.

 

 
 [186] 
 
S. K. Sampat, P. Banerjee, Y. Yang, and C. Baral, “Learning action-effect dynamics for hypothetical vision-language reasoning task,” arXiv preprint arXiv:2212.03866 , 2022.

 

 
 [187] 
 
R. Moraffah, M. Karami, R. Guo, A. Raglin, and H. Liu, “Causal interpretability for machine learning-problems, methods and evaluation,” ACM SIGKDD Explorations Newsletter , vol. 22, no. 1, pp. 18–33, 2020.

 

 
 [188] 
 
Y. Huang, Z. Fu, and C. L. Franzke, “Detecting causality from time series in a machine learning framework,” Chaos: An Interdisciplinary Journal of Nonlinear Science , vol. 30, no. 6, 2020.

 

 
 [189] 
 
M. Li and K. Liu, “Causality-based attribute weighting via information flow and genetic algorithm for naive bayes classifier,” IEEE Access , vol. 7, pp. 150 630–150 641, 2019.

 

 
 [190] 
 
I. Blecic, A. Cecchini, and G. A. Trunfio, “A decision support tool coupling a causal model and a multi-objective genetic algorithm,” Applied Intelligence , vol. 26, pp. 125–137, 2007.

 

 
 [191] 
 
L. M. de Campos and J. F. Huete, “Approximating causal orderings for bayesian networks using genetic algorithms and simulated annealing,” in Proceedings of the Eighth IPMU Conference , vol. 1, no. 2000, 2000.

 

 
 [192] 
 
T. Deleu, A. Góis, C. Emezue, M. Rankawat, S. Lacoste-Julien, S. Bauer, and Y. Bengio, “Bayesian structure learning with generative flow networks,” in Uncertainty in Artificial Intelligence . PMLR, 2022, pp. 518–528.

 

 
 [193] 
 
N. Pawlowski, D. Coelho de Castro, and B. Glocker, “Deep structural causal models for tractable counterfactual inference,” Advances in Neural Information Processing Systems , vol. 33, pp. 857–869, 2020.

 

 
 [194] 
 
P. Darondeau and P. Degano, “Causal trees,” in International Colloquium on Automata, Languages, and Programming . Springer, 1989, pp. 234–248.

 

 
 [195] 
 
M. Kocaoglu, C. Snyder, A. G. Dimakis, and S. Vishwanath, “Causalgan: Learning causal implicit generative models with adversarial training,” arXiv preprint arXiv:1709.02023 , 2017.

 

 
 [196] 
 
I. Shpitser and J. Pearl, “Identification of joint interventional distributions in recursive semi-markovian causal models,” in Proceedings of the National Conference on Artificial Intelligence , vol. 21, no. 2. Menlo Park, CA; Cambridge, MA; London; AAAI Press; MIT Press; 1999, 2006, p. 1219.

 

 
 [197] 
 
M. Yang, F. Liu, Z. Chen, X. Shen, J. Hao, and J. Wang, “Causalvae: Disentangled representation learning via neural structural causal models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2021, pp. 9593–9602.

 

 
 [198] 
 
G. Wunsch, F. Russo, and M. Mouchart, “Do we necessarily need longitudinal data to infer causal relations?” Bulletin of Sociological Methodology/Bulletin de Méthodologie Sociologique , vol. 106, no. 1, pp. 5–18, 2010.

 

 
 [199] 
 
J. Berrevoets, K. Kacprzyk, Z. Qian, and M. van der Schaar, “Causal deep learning,” arXiv preprint arXiv:2303.02186 , 2023.

 

 
 [200] 
 
D. Heckerman, “A bayesian approach to learning causal networks,” arXiv preprint arXiv:1302.4958 , 2013.

 

 
 [201] 
 
L. Lei and E. J. Candès, “Conformal inference of counterfactuals and individual treatment effects,” Journal of the Royal Statistical Society Series B: Statistical Methodology , vol. 83, no. 5, pp. 911–938, 2021.

 

 
 [202] 
 
P. Bühlmann, J. Peters, and J. Ernest, “Cam: Causal additive models, high-dimensional order search and penalized regression,” The Annals of Statistics , 2014.

 

 
 [203] 
 
V. Balazadeh Meresht, V. Syrgkanis, and R. G. Krishnan, “Partial identification of treatment effects with implicit generative models,” Advances in Neural Information Processing Systems , vol. 35, pp. 22 816–22 829, 2022.

 

 
 [204] 
 
C. Louizos, U. Shalit, J. M. Mooij, D. Sontag, R. Zemel, and M. Welling, “Causal effect inference with deep latent-variable models,” Advances in neural information processing systems , vol. 30, 2017.

 

 
 [205] 
 
Y. Luo, J. Peng, and J. Ma, “When causal inference meets deep learning,” Nature Machine Intelligence , vol. 2, no. 8, pp. 426–427, 2020.

 

 
 [206] 
 
H. Kim, S. Shin, J. Jang, K. Song, W. Joo, W. Kang, and I.-C. Moon, “Counterfactual fairness with disentangled causal effect variational autoencoder,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 35, no. 9, 2021, pp. 8128–8136.

 

 
 [207] 
 
V. Terziyan and O. Vitko, “Causality-aware convolutional neural networks for advanced image classification and generation,” Procedia Computer Science , vol. 217, pp. 495–506, 2023.

 

 
 [208] 
 
Y. Wang, “Fuzzy causal inferences based on fuzzy semantics of fuzzy concepts in cognitive computing,” WSEAS Transactions on Computers , vol. 13, pp. 430–441, 2014.

 

 
 [209] 
 
H. S. Kim and K. C. Lee, “Fuzzy implications of fuzzy cognitive map with emphasis on fuzzy causal relationship and fuzzy partially causal relationship,” Fuzzy Sets and Systems , vol. 97, no. 3, pp. 303–313, 1998.

 

 
 [210] 
 
Y. Miao and Z.-Q. Liu, “On causal inference in fuzzy cognitive maps,” IEEE transactions on Fuzzy Systems , vol. 8, no. 1, pp. 107–119, 2000.

 

 
 [211] 
 
Q. Zhang and W. J. Doll, “The fuzzy front end and success of new product development: a causal model,” European Journal of Innovation Management , vol. 4, no. 2, pp. 95–112, 2001.

 

 
 [212] 
 
W. Chen, “A quantitative fuzzy causal model for hazard analysis of man–machine-environment system,” Safety science , vol. 62, pp. 475–482, 2014.

 

 
 [213] 
 
C.-J. Lin and W.-W. Wu, “A causal analytical method for group decision-making under fuzzy environment,” Expert systems with applications , vol. 34, no. 1, pp. 205–213, 2008.

 

 
 [214] 
 
B. Kosko, “Fuzzy cognitive maps,” International journal of man-machine studies , vol. 24, no. 1, pp. 65–75, 1986.