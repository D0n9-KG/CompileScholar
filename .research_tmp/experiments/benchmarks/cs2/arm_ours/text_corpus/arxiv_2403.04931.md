A Survey on Human-AI Collaboration with Large Foundation Models 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2403.04931v4 [cs.AI] 22 Aug 2026 
 
 

# A Survey on Human-AI Collaboration with Large Foundation Models DOI:  XXXXXXX.XXXXXXX CCS:  Computing methodologies Artificial intelligence CCS:  Human-centered computing Human computer interaction (HCI) Journal:  TIST 8 DOI:  10.1145/3841472 

 
 
 Vanshika Vats
 
 Note:  Shared First Authorship
 
 email: vvats@ucsc.edu 
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Marzia Binta Nizam
 
 email: manizam@ucsc.edu 
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Minghao Liu
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Ziyuan Wang
 
 Note:  Equal Contribution
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Richard Ho
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Mohnish Sai Prasad
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Vincent Titterton
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Sai Venkat Malreddy
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Riya Aggarwal
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Yanwen Xu
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Lei Ding
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Jay Mehta
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Nathan Grinnell
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Li Liu
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Sijia Zhong
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Devanathan Nallur Gandamani
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Xinyi Tang
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Rohan Ghosalkar
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Celeste Shen
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Rachel Shen
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Nafisa Hussain
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 , 
 Kesav Ravichandran
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 and 
 James Davis
 
 Affiliation:  University of California, Santa Cruz 
,  Santa Cruz 
,  California 
,  USA 
 
 2026© , 2026; 

 Abstract. 
 
 As the capabilities of artificial intelligence (AI) continue to expand rapidly, Human-AI (HAI) Collaboration, combining human intellect and AI systems, has become pivotal for advancing problem-solving and decision-making processes. The advent of Large Foundation Models (LFMs) has greatly expanded its potential, offering unprecedented capabilities by leveraging vast amounts of data to understand and predict complex patterns. At the same time, realizing this potential responsibly requires addressing persistent challenges related to safety, fairness, and control. This paper reviews the crucial integration of LFMs with HAI, highlighting both opportunities and risks. We structure our analysis around: human-guided model development, collaborative design principles, ethical and governance frameworks, and applications in high-stakes domains. Our review shows that successful HAI systems are not the automatic result of stronger models but the product of careful, human-centered design. By identifying key open challenges, this survey aims to give insight into current and future research that turns the raw power of LFMs into partnerships that are reliable, trustworthy, and beneficial to society.

 
 
 
 Keywords:  Human-AI Collaboration, Artificial Intelligence, Large Foundation Models, Large Language Models
 
 

## 1. Introduction

 
 People have long been fascinated by the idea of creating machines that think like humans. One notable example from the 1770s is the "Mechanical Turk," a machine that appeared to play chess autonomously but was, in fact, operated by a person concealed inside it ( mech_turk ) . This early effort, while not "Artificial Intelligence" (AI) in the modern sense, reflects the enduring fascination with creating technology that can mimic or complement human cognitive abilities. The formal realization of integrating human-like intelligence into machines peaked in 1956 at Dartmouth College, USA, marking the official birth of Artificial Intelligence as a research field ( Russell2021 ) . As AI technology has advanced and become more prevalent over the last decade, researchers have identified limitations within purely automated systems ( mittelstadt2016 ; goodfellow2015 ; rodrigues2020 ) leading to renewed focus on augmenting AI with human expertise, aiming to utilize the best of both worlds (Fig. 1(a) ). This approach seeks to enhance AI applications by combining automation with human judgment, creating a more symbiotic collaboration between human and artificial intelligence.

 
 
 
 
 
 (a) 
 
 
 (b) 
 
 Figure 1 . A graphical depiction of the number of relevant articles submitted from 2014-2025. (a) illustrates the increasing engagement of research communities in collaborations between humans and AI over the years. (b) observes a notably sharp increase in submissions related to human-AI and large-scale models attributed to the recent emergence of large models. Two charts show the growth of Human-AI research from 2014 to 2025. Two side-by-side bar charts plot the number of articles by year from 2014 to 2025. The left chart shows Human-AI research increasing gradually through 2020 and then sharply after 2021, reaching nearly 16,000 articles in 2025. The right chart shows research combining Human-AI topics with large models remaining limited until 2022, followed by rapid growth in 2023-2025 and more than 11,000 articles in 2025.
 
 
 
 Navigating the rapidly evolving AI landscape, the integration of human cognition with Large Foundation Models (LFMs), including Large Language Models (LLM) ( Tom2020 ; touvron2023llama ) and Large Vision Models (LVM) ( kirillov2023segany ; pmlr-v139-radford21a ; jia2022vpt ) , has initiated an exciting shift. These models are pre-trained on vast web-scale datasets and act as "foundations" before being fine-tuned for specific tasks. This has opened new avenues for collaborative problem-solving and decision-making. When humans add domain insight, ethics, and creativity to these generic large models, they return rapid pattern-finding and specialized outputs at scale. In turn, AI amplifies human capabilities by processing data at large scales, offering insights, and augmenting decision-making. This interaction paves the way to creating an advanced form of HAI collaboration. In the LFM era, this collaboration spans both build-time settings, where humans shape what models learn, and run-time settings, where humans and AI systems divide initiative, control, and responsibility during task execution.

 
 
 For this survey paper, we formally define the term ‘Human-AI Collaboration’. Human-AI (HAI) collaboration is a cooperative partnership in which humans and AI systems coordinate their complementary strengths, exchange information (and sometimes control), and pursue a shared goal, without the tight interdependence and joint accountability ( kamar2016directions ; shneiderman2020human ; WilsonDaugherty2018HBR ) . Each partner contributes what it does best and iteratively refines the joint output through feedback. This paper undertakes an extensive survey of HAI collaboration, analyzing the complex interactions between human agents and sophisticated pre-trained models across various fields. Our study aims to uncover the progress made, confront the challenges, and understand the implications of this evolving partnership. To synthesize this literature, we use recurring collaboration patterns as a guiding lens: human steering of AI, AI augmentation of human work, mixed-initiative collaboration, dynamic delegation or control handoff, and emerging supervisory orchestration in agentic systems.

 
 
 
 
 
 (a) 
 
 
 (b) 
 
 Figure 2 . A representation of the scope of this survey. (a) We cover four broad categories - Human Guided AI Development, Design Principles for HAI Collaboration, Ethics and Governance, and their Applications - in the human-AI domain, along with the recent developments in them with the help of LFMs. The overlap between the four petals illustrates the fact that the four categories are not mutually exclusive; the articles may belong to two or more categories at the same time. (b) The heatmap shows a general trend of the number of articles co-existing within subcategories, including the articles discussing large foundation models within each subject area. The overlap between the applications is not analyzed, given their specific and varied nature. HAI : Human-AI, HITL : Human-in-the-Loop. The survey scope is shown through four overlapping categories and a heatmap of article overlap. Panel (a) places Human-AI and Large Foundation Models at the center of four overlapping areas: Human-Guided AI Development at the top, Ethics and Governance on the left, Applications on the right, and Design Principles for HAI Collaboration at the bottom. Their overlap indicates that papers may address multiple areas. Panel (b) is a heatmap whose rows and columns represent survey subcategories, including human-in-the-loop learning, evaluation, interface design, collaboration, system integration, fairness, autonomy, labor, privacy, accountability, and regulation. Darker cells indicate more articles shared by a pair of subcategories, while the first column highlights articles involving large foundation models.
 
 
 

### 1.1. Scope of the Survey

 
 Our survey focuses on the articles describing the developments of human-AI collaboration through the years and how the introduction of large models is reforming the field. The search for articles was conducted on Google Scholar focusing on the years between 2014-2025, using keywords such as "Human-AI", and "Human-AI collaboration". For research focusing on the collaboration between humans and AI involving large models, additional keywords, including "large models", "foundation models", "large language models", and "large vision models" were utilized. Because the field has rapidly shifted toward agentic and multi-agent AI systems, we also include a targeted update of selected 2025-early 2026 works on tool-use agents, multi-agent orchestration, computer-use agents, and human supervision of long-horizon AI workflows. The content was then distributed among the authors to filter based on the theme of our review to be structured into human-guided AI development, design principles for HAI collaboration, ethics and governance, and finally, applications. Some of the selected papers fall into two or more categories, as illustrated in Fig. 2 . Since human-AI collaboration and large models have only recently gained a lot of interest from the research community given the recent popularity of large foundation models (Fig. 1(b) ), some portion of studies cited in this survey could also be from arXiv preprints. We then sample the articles according to the structure desired for this survey, i.e., primarily covering Human-AI collaboration and its large foundation model counterparts in better model training, effective human-AI joint systems, safe and secure HAI collaboration, and their applications (Fig. 2(a) ). Although not systematic, this review captures the most visible work across AI, HAI, and collaboration.

 
 
 In order to show support for the theme of our survey article, we try to implement HAI collaboration with the help of large language models to prepare this review. For the human part, the authors survey the existing literature about human-AI collaboration, collect and filter a subset of articles according to the desired structure, organize the survey into relevant sections, study submission statistics, put together tables and visualizations, and finally, prepare the first complete draft of the manuscript. Using the AI part in human-AI collaboration, we ask ChatGPT ( OpenAI2023ChatGPT ) for its feedback regarding the language flow of each subsection, placement of the citations, and revising the language wherever necessary. This helps to ease in identifying the missing elements and for the authors to make necessary revisions. The final manuscript is, therefore, a product of human and AI collaboration.

 
 
 Figure 3 . This figure summarizes the lifecycle of Human–AI collaboration with LFMs. Section 2 covers build-time human guidance of models; Section 3 organizes run-time collaboration into recurring patterns such as AI augmentation, mixed initiative, control handoff, and emerging supervisory orchestration; Section 5 shows how these patterns appear across application domains. Section 4 acts as a cross-cutting governance layer, while Section 6 identifies open challenges that arise across the lifecycle. The Human-AI collaboration lifecycle connects model development, run-time collaboration, applications, governance, and open challenges. The central flow moves from Section 2, Human-Guided Model Development, to Section 3, Design Principles for Human-AI Collaboration, and then to Section 5, Applications. These stages respectively represent building the AI partner, working with the AI partner, and applying collaboration across domains. Section 4 forms an enclosing governance layer around all three stages. A dashed post-deployment monitoring path feeds application experience back into model development and collaboration design. Section 6 appears below the lifecycle and represents open challenges and future research directions spanning all stages.
 
 
 
 Table 1. A tabular representation of the four broader focus topics covered in this survey, with their relevant subtopics. Each category contains its cited articles for the ease of reader reference. 
 
 
 
 
 
 Human-AI Topics 
 | 
 
 
 
 
 
 Subtopics 
 | 
 
 
 Articles Cited 
 | 

 | 

 
 
 
 
 
 Section 2 : Human-Guided Model Development 
 | 
 
 
 
 
 
 HITL and Active-Learning 
 | 
 
 
 ( Whittlestone2019 ) , ( hu2023 ) , ( Memmert2023 ) , ( Settles2009 ) , ( Pries2023 ) , ( Lu2022 ) , ( Bao2018 ) , ( Lertvittayakumjorn2020 ) , ( Yao2023 ) 
 | 

 
 
 
 Human-Guided Objective Shaping 
 | 
 
 
 ( christiano2017 ) , ( ouyang2022 ) , ( stiennon2020 ) , ( rafailov2023 ) , ( bai2022 ) , ( lee2023rlaif ) , ( kaelbling1996reinforcement ) , ( nakano2021 ) , ( align-anything-200k ) , ( vodrahalli2025canonical ) , ( hu2023 ) , ( Mirchandani2023 ) , ( Wei2022 ) , ( Zha2023 ) , ( zhang2021 ) , ( johnson2019no ) , ( McNeese2021 ) , ( Zhang2025MMRLHF ) 
 | 

 
 
 
 Human Evaluation and Alignment Metrics 
 | 
 
 
 ( bansal2019 ) , ( Jianlong2019 ) , ( align-anything-200k ) , ( vodrahalli2025canonical ) , ( clark2021 ) , ( Chhibber2022 ) , ( hu2023 ) , ( Tabrez2020 ) , ( lu2023 ) , ( uchendu2023 ) , ( liu2025survey ) 
 | 

 | 

 
 
 
 Section 3 : Design Principles for Human-AI Collaboration 
 | 
 
 
 
 
 
 Interaction and Interface Design 
 | 
 
 
 ( zhang2021 ) , ( Hauptman2023 ) , ( Bryan2023 ) , ( Qing2023 ) , ( Mingming2023 ) , ( Tongshuang2022 ) , ( Kozlowski2006 ) , ( Salas2017 ) , ( Zhang2023 ) , ( Chhibber2022 ) , ( Clark2016 ) , ( wang2019human ) , ( Bosch2019 ) , ( Flathmann2021 ) , ( wei2023 ) , ( Lemaignan2017 ) 
 | 

 
 
 
 Collaboration Patterns and Role Allocation 
 | 
 
 
 ( wei2023 ) , ( Tongshuang2022 ) , ( Pflanzer2022 ) , ( dubey2020 ) , ( liu2023tag ) , ( ngo2023tag ) , ( Zhang2023 ) , ( Gopinath2022 ) , ( Munyaka2023 ) , ( rafailov2023 ) , ( nakano2021 ) , ( hu2023 ) , ( ouyang2022 ) , ( ziegler2019 ) , ( Jianlong2019 ) , ( henry2022 ) , ( Lv2021 ) , ( wang2023 ) , ( Veselovsky2023 ) , ( pal2023 ) , ( Memmert2023 ) , ( ahmad2023towards ) , ( Fuchs2023 ) , ( bansal2021does ) , ( Chakraborti2017 ) , ( Bosch2019 ) , ( liu2023humans ) , ( Neerincx2018 ) , ( Nikolaidis2017 ) , ( Zhao2022 ) , ( Kaptein2016 ) , ( Rastogi2023 ) , ( Sharma2023 ) , ( Chakrabarty2023 ) , ( Sharifi2022 ) , ( fourney2024magenticone ) , ( motwani2024malt ) , ( yu2025table ) , ( vats2026guideline ) , ( schoembs2026conversation ) , ( masters2025orchestrating ) , ( yao2024tau ) , ( xie2024osworld ) , ( abhyankar2025osworld ) , ( xu2026theagentcompany ) , ( jiang2025medagentbench ) 
 | 

 
 
 
 System Integration and Evaluation 
 | 
 
 
 ( Weisz2021 ) , ( Chhibber2022 ) , ( Bau2019 ) , ( wang2023 ) , ( Eloundou2023 ) , ( harrer2023 ) , ( McNeese2021 ) , ( lyons2019 ) , ( Hou2023 ) , ( bansal2021does ) , ( Tongshuang2022 ) , ( Veselovsky2023 ) , ( henry2022 ) , ( liu2023humans ) , ( Munyaka2023 ) , ( Memmert2023 ) , ( chang2023 ) 
 | 

 | 

 
 
 
 Section 4 : Ethical, Societal and Governance Aspects 
 | 
 
 
 
 
 
 Fairness in Collaborative Decision-Making 
 | 
 
 
 ( Ben2019 ) , ( Stahl2021 ) , ( Nicholas2023 ) , ( Chhibber2022 ) , ( Morrison2023 ) , ( kirk2021 ) , ( huang2023bias ) , ( kotek2023 ) , ( meyer2023chatgpt ) , ( li2023bias ) , ( gallegos2023bias ) , ( ohi2024bias ) , ( bi2023bias ) , ( ling2024applicants ) 
 | 

 
 
 
 Empowering Human Autonomy and Well-Being 
 | 
 
 
 ( Chhibber2022 ) , ( Konstantis2023 ) , ( pal2023 ) , ( despotovic2024 ) , ( wang2023 ) , ( woodruff2023 ) 
 | 

 
 
 
 Collaborative Labor Dynamics 
 | 
 
 
 ( Hemmer2023 ) , ( CHOWDHURY202231 ) , ( Chhibber2022 ) , ( pal2023 ) , ( Eloundou2023 ) , ( Purdy2016 ) , ( despotovic2024 ) , ( wang2023 ) , ( walkowiak2023 ) 
 | 

 
 
 
 Data privacy and Security 
 | 
 
 
 ( Ezer2019 ) , ( Yin2019 ) , ( Kaissis2020 ) , ( Admin2022 ) , ( Lepri2021 ) , ( Hacker2023 ) , ( ullah2023 ) , ( gupta2023 ) , ( de2023 ) , ( thapa2023 ) , ( sebastian2023 ) 
 | 

 
 
 
 Building Trust and Shared Accountability 
 | 
 
 
 ( Hou2023 ) , ( Pflanzer2022 ) , ( caldwell2022agile ) , ( choudhury2024large ) , ( kim2023help ) , ( bansal2019 ) 
 | 

 
 
 
 Policy and Regulation 
 | 
 
 
 ( jobin2019 ) , ( cath2018 ) , ( Stahl2021 ) , ( Bender2021 ) , ( Chinonso2023 ) , ( mit2023law ) , ( gdpr2023law ) 
 | 

 | 

 
 
 
 Section 5 : Applications 
 | 
 
 
 
 
 
 Healthcare 
 | 
 
 
 ( henry2022 ) , ( Bienefeld2023 ) , ( Memmert2022 ) , ( Carrie2019 ) , ( McKinney2020 ) , ( BUDD2021 ) , ( CHOUDHURY2022 ) , ( lyu2023 ) , ( kung2023 ) , ( johnson2023 ) , ( strong2024deferral ) , ( biswas2024clinicaldoc ) 
 | 

 
 
 
 Autonomous Vehicles 
 | 
 
 
 ( Atakishiyev2021 ) , ( Lv2021 ) , ( Jianlong2019 ) , ( Fuchs2023 ) , ( yang2024 ) , ( park2024 ) , ( cui2024drive ) , ( tian2024critical ) , ( wang2023empowering ) , ( wen2023 ) , ( wang2023drive ) 
 | 

 
 
 
 Surveillance and Security 
 | 
 
 
 ( killcrece_2003 ) , ( Pazho2023 ) , ( Hauptman2023 ) , ( Guo2018 ) , ( chen2023 ) , ( jain2023 ) , ( baruwalchhetri2024alertfatigue ) , ( oliver2024carbonfilter ) 
 | 

 
 
 
 Games 
 | 
 
 
 ( MarioKart8Deluxe2017 ) , ( frans2021 ) , ( xu2023werewolf ) , ( sobieszek2022 ) , ( akata2023 ) , ( sidji2024codenames ) , ( todd2023 ) , ( vartinen2022 ) , ( hu2023 ) , ( white2024communicate ) 
 | 

 
 
 
 Education 
 | 
 
 
 ( nwana1990 ) , ( vanlehn2011 ) , ( whitaker2013 ) , ( kasinathan2017 ) , ( eicher2018 ) , ( hartle2019 ) , ( dikli2006 ) , ( extance2023 ) , ( milano2023 ) , ( rose2023a ) , ( hellas2023 ) , ( chang2023 ) , ( kong2025synergy ) , ( schotter2025spiral ) 
 | 

 
 
 
 Accessibility 
 | 
 
 
 ( kumar2022 ) , ( Ozarkar2020 ) , ( Khan2020 ) , ( Wen2021 ) , ( Ghazal2021 ) , ( taheri2023 ) , ( gadiraju2023 ) , ( brilli2024airis ) , ( tokmurziyev2025llmglasses ) 
 | 

 | 

 

 
 
 

### 1.2. Outline of the Survey

 
 The survey follows the lifecycle summarized in Fig. 3 , moving from human-guided model development to run-time collaboration, governance, applications, and open challenges. In each segment of our study, we commence by delineating traditional methodologies employed in human-AI collaboration, subsequently delving into the contributions of extensive pre-trained foundation models in the fulfillment of these tasks (Table 1 ). Section 2 explores the involvement of incorporating human expertise into the AI model training cycle, fostering a cooperative relationship between humans and AI in active learning scenarios, enhancing learning through human feedback, and involving human experts in the thorough evaluation of machine learning models ( hu2023 ; Settles2009 ; ziegler2019 ) . Section 3 shifts to run-time collaboration, where the trained model becomes a partner whose interface, role, initiative, and control boundaries must be designed ( Lemaignan2017 ; Mingming2023 ) .

 
 
 A major concern in Human-AI collaboration is the safety, security, and trustworthiness of AI systems. Section 4 of our comprehensive analysis covers multiple dimensions: mitigating algorithmic biases to uphold fairness, assessing the impact of Human-AI collaboration on workers’ autonomy, well-being, and job satisfaction, examining economic repercussions on employment and wages, addressing data privacy and security concerns, building trust in AI systems, and navigating the legal and regulatory frameworks governing Human-AI interactions ( Konstantis2023 ; Hemmer2023 ; Braun2021 ) . These factors are integral to creating an ethical and responsible environment for AI’s integration into human-centric workflows. Section 5 explores the broad spectrum of Human-AI collaboration applications across various sectors. We investigate the unique challenges and opportunities in fields such as healthcare ( Bienefeld2023 ) , autonomous vehicles, surveillance systems ( Pazho2023 ; Guo2018 ) , gaming ( frans2021 ) , education, and accessibility. Understanding the specific characteristics and benefits of Human-AI collaboration in these areas is key to influencing its future direction and maximizing its societal impact. Finally, Section 6 summarizes the core findings and identifies key open challenges and future research directions, providing a roadmap for building the next generation of effective, fair, and trustworthy human-AI partnerships.

 
 
 
 

## 2. Human-Guided AI Model Development: Building the AI Partner 

 
 Figure 4 . The three-phase iterative framework for human-guided model development - comprising data control, model optimization, and evaluation, as described by Maadi et al. ( Maadi2021 ) . An iterative framework links data control, model optimization, and evaluation. Three stages of Data Control, Model Optimization, and Evaluation are arranged from left to right. Forward arrows show data being used to optimize a model and the resulting model being evaluated. A return arrow from Evaluation to Data Control closes the loop, indicating that evaluation results inform the next round of data selection or correction. 
 
 
 This section examines build-time collaboration, where humans shape LFMs through data, objectives, and evaluation before the model is deployed as a collaborative partner. We adapt the three-phase view of Maadi et al. ( Maadi2021 ) , data control → \rightarrow model optimization → \rightarrow evaluation (Fig. 4 ), to review the points where human insight most clearly improves an FM:

 
 • 
 
 Sec. 2.1 , Human-in-the-Loop Data Curation Active Learning: People curate or label the right examples and, for LFMs, craft or filter instruction prompts that guide later fine-tuning.

 

 • 
 
 Sec. 2.2 , Human-Guided Objective Shaping: Preference-based methods such as RLHF and DPO let human feedback reshape the model’s reward or loss surface during optimization.

 

 • 
 
 Sec. 2.3 , Human Evaluation and Alignment Metrics: Domain experts and end-users judge outputs, supplying both task scores and subjective signals of trust, fairness, and usability, which feed into the next iteration of the HAI collaboration pipeline.

 

 
 
 

### 2.1. Human-in-the-Loop Data and Active-Learning

 
 Human-in-the-Loop (HITL) methods exemplify human-AI collaboration by combining human judgment with automated efficiency. Humans guide model design by bringing ethical insight beyond technical metrics (Whittlestone et al. ( Whittlestone2019 ) ), while AI accelerates data processing and suggests patterns that might not be easy to catch for the time-constrained annotators. During data construction and fine-tuning, information flows bidirectionally. For example, NLP-based directives like InstructRL, as proposed by Hu and Sadigh ( hu2023 ) , steer AI toward more interpretable, aligned behaviors, and AI-generated prompts enrich human brainstorming, sparking novel ideas ( Memmert2023 ) . The diverse roles of humans, ranging from providing ethical oversight in the design phase to contributing to quality control and data labeling during model training and execution, are crucial for optimizing AI performance.

 
 

#### 2.1.1. Human-AI Collaboration with Active Learning 

 
 Unlike traditional machine learning concepts, active learning involves an iterative process where a model selectively identifies data requiring labeling, optimizing system performance with minimal training data. Particularly, this approach is crucial for fine-tuning large pre-trained foundation models that require substantial labeled data for specific user scenarios. Active learning efficiently utilizes human expertise to pinpoint areas of uncertainty, enabling more targeted training with less annotation effort ( Settles2009 ) . The active learning workflow begins with an unlabeled dataset and a pre-trained model. The model predicts labels for each sample, outputting confidence levels (Fig. 5 ). When predictions fall below a quality threshold, human annotators step in for manual annotation. This iterative process of re-training the model with new labeled data continues until satisfactory confidence levels are achieved. Humans and AI thus collaborate in efficient data labeling and continuous model improvement.

 
 
 Figure 5 . A visual representation of an active learning workflow. If the model predictions do not pass the required confidence value thresholds required for a task, the data is sent to humans for manual annotations, further improving the model by retraining it with the now-labeled data. An active-learning workflow routes uncertain predictions to human annotators and uses their labels to retrain the model. Unlabeled data enters a pre-trained model, which produces a prediction that is checked against a confidence threshold. If the prediction passes, it is accepted. If it does not pass, the sample is sent to human annotators for manual annotation. The resulting labeled samples are added to the labeled dataset, which is then used to retrain the pre-trained model, forming an iterative improvement loop. 
 
 
 Recent research in this area has introduced novel methods to leverage human expertise effectively. For instance, Pries et al. ( Pries2023 ) propose a method for efficient data labeling using precise distance measurements to filter data samples, streamlining the presentation of data to experts. Lu et al. ( Lu2022 ) discuss human-guided interventions for continuous model improvement, focusing on dissociating biases. Bao et al. ( Bao2018 ) explore the transformation of human-annotated rationales into continuous attention mechanisms, enhancing the learning of domain-invariant representations. Additionally, Lertvittayakumjorn et al. ( Lertvittayakumjorn2020 ) present a generalizable approach for debugging deep text classification models with human input, applicable to larger models. Finally, Yao et al. ( Yao2023 ) propose an explainable-generation active learning framework for simpler tasks, potentially improving the sampling process in data augmentation.

 

 
 
 Taken together, research in this area shows a shift in the human role in data curation. Traditional active learning balances annotation cost against model performance, but with Large Foundation Models, humans act less as mass annotators and more as targeted teachers during fine-tuning. This shift brings a difficult trade-off: small, carefully chosen datasets must be diverse enough to reduce, not reinforce, the biases already present in the model. A key open challenge remains to design bias-aware active learning methods that identify not only areas of model uncertainty but also regions where pre-trained knowledge is skewed or incomplete.

 
 
 
 

### 2.2. Human-Guided Objective Shaping

 
 After curating the data that an LFM sees, collaborators can intervene a second time by reshaping the objective it tries to optimize. This extends human-guided model development beyond labeling examples: humans express preferences, rankings, or critiques over model outputs, indicating which behaviors should be preferred when multiple responses are plausible. These signals are then converted into a learnable reward or direct policy update. The canonical pipeline, Reinforcement Learning from Human Feedback (RLHF) ( christiano2017 ; ouyang2022 ; stiennon2020 ) , trains a reward model on pairwise preference judgements and then fine-tunes the policy to maximize that reward, producing assistants that are measurably more helpful and safer. Simpler, gradient-based alternatives such as Direct Preference Optimization (DPO) ( rafailov2023 ) and Constitutional AI ( bai2022 ) bypass reinforcement yet achieve similar alignment by treating human or rule-based critiques as a differentiable signal. This section surveys these preference-based algorithms, the types of human feedback they require, and open challenges in keeping objectives both scalable and faithful to human intent.

 
 

#### 2.2.1. RLHF Pipeline. 

 
 Typically, the RLHF approach involves three stages ( ouyang2022 ) : initially, the pre-trained foundation model generates candidate responses; human annotators then evaluate these outputs, providing explicit preference-based feedback; subsequently, a reward model is trained on these human-generated evaluations. Finally, policy optimization methods, often proximal policy optimization (PPO) or direct preference optimization (DPO) ( rafailov2023 ) , fine-tune the foundation model to maximize the learned reward. This iterative human–AI feedback loop enables continuous and targeted improvements in model alignment.

 
 
 

#### 2.2.2. Collaborative Advantages. 

 
 RLHF’s reliance on human evaluation significantly enhances the alignment of models with ethical and functional expectations. Human annotators can actively identify and mitigate harmful biases or unsafe outputs early in the optimization process, producing fairer and more trustworthy AI systems ( lee2023rlaif ) . Additionally, the iterative nature of human feedback facilitates rapid adaptation to evolving domain-specific or user-specific preferences, overcoming the rigidity inherent in traditional supervised learning methods. Transparency is also notably improved, as human-generated feedback and reward insights provide explicit visibility into the rationale behind model decision-making ( kaelbling1996reinforcement ) . The adaptability of RLHF frameworks to user and domain-specific contexts further enhances their practical effectiveness, enabling personalized and contextually significant AI behaviors.

 
 
 Despite these advantages, implementing RLHF involves several practical challenges. Ensuring consistency and accuracy of human feedback, scaling RLHF effectively to large datasets, and aligning human insights with algorithmic processes are key considerations that must be addressed to fully leverage RLHF’s potential ( lee2023rlaif ) .

 
 
 

#### 2.2.3. Extensions of Preference-Based Alignment 

 
 Recent methodological advances have expanded preference-based alignment. OpenAI’s InstructGPT ( ouyang2022 ) introduced a structured three-phase training process comprising supervised policy training, reward model training with human feedback, and subsequent reinforcement learning-based policy optimization. Similarly, Direct Preference Optimization (DPO) simplifies the training by directly mapping human feedback into policy improvements, thereby reducing complexity and enhancing the quality of outputs ( rafailov2023 ) . A recent survey of DPO further shows that preference optimization is broadening beyond the basic RLHF-free objective, with variants organized around data strategy, learning frameworks, constraint mechanisms, and model properties ( liu2025survey ) . Further illustrating RLHF’s potential, Nakano et al. ( nakano2021 ) introduced web browser-assisted agents trained via RLHF, demonstrating significant improvements in real-world usability. Recent studies have further expanded RLHF frameworks through innovative approaches. Ji et al. ( align-anything-200k ) introduced a dataset enabling models to integrate multimodal human feedback, spanning text, images, audio, and video, thereby significantly enriching the interaction between humans and AI. Additionally, Vodrahalli et al. ( vodrahalli2025canonical ) proposed a canonical basis of human preferences, identifying a compact yet expressive set of fundamental categories that efficiently represent diverse human judgments, potentially reducing annotation overhead and enhancing model interpretability.

 
 
 Extending beyond language models, RLHF methodologies have influenced fields like robotics and interactive decision-making systems. Hu and Sadigh’s InstructRL ( hu2023 ) employs natural-language instructions to effectively guide robotic actions, underscoring the versatility of RLHF approaches. Additionally, research by Mirchandani et al. ( Mirchandani2023 ) highlights the utility of large language models (LLMs) in robotics, leveraging their in-context learning abilities to facilitate complex sequential tasks. Chain-of-thought prompting techniques ( Wei2022 ) and correction-based learning frameworks such as DROC ( Zha2023 ) further exemplify RLHF’s expanding scope.

 
 
 Nevertheless, RLHF models still face some hurdles, notably in bridging communication barriers between humans and AI. Studies by Zhang et al. ( zhang2021 ) and Johnson and Vera ( johnson2019no ) emphasize these challenges, highlighting the need for more sophisticated human-AI communication frameworks. McNeese et al. ( McNeese2021 ) further underscore the limitations of current RLHF models in managing real-world complexities, particularly regarding nuanced human communication and the ability to generalize from simulated environments to real-world settings.

 

 
 
 To summarize, the work on objective shaping marks a shift from programming explicit goals to learning implicit human values through feedback. RLHF and its variants demonstrate that human preference signals can reliably steer models toward safer and more useful behaviors. However, the practical limits of noisy, costly, and potentially narrow annotation remain. Recent advances ( vodrahalli2025canonical ; Zhang2025MMRLHF ) point toward concrete solutions: datasets that capture multimodal human feedback (text, images, audio, video) and frameworks that distill a canonical basis of preferences reduce annotation overhead while broadening coverage of diverse values. Together, these directions suggest that scalable, bias-resilient objective shaping is achievable, not by collapsing human preferences into a single worldview, but by structuring them into compact, expressive, and extensible feedback spaces.

 
 
 Figure 6 . Evaluation training strategies based on Clark et al. ( clark2021 ) . They employ training methods by giving example and comparison tests, along with their correct answers. This is followed by an explanation of why the answer is correct. This way, the evaluators gradually learn about how to distinguish between human and machine-generated texts. Two evaluator-training tasks teach people to distinguish human-written and machine-written text. The figure begins with two example texts, one labeled as human-written and the other as machine-written. In Example Training, an evaluator sees one text and selects among four source judgments ranging from definitely human to definitely machine; the correct answer is shown as definitely machine. In Comparison Training, the evaluator sees both texts and identifies which one is machine-generated; Text 2 is marked correct. Both tasks are followed by an explanation of cues such as repetitive or contradictory writing.
 
 
 
 
 

### 2.3. Human Evaluation and Alignment Metrics

 
 Human evaluation is the final collaborative safeguard that ensures foundation models satisfy both task requirements and human expectations. Rather than serving as a single ‘final report card’ human evaluation forms a feedback loop: people surface errors, voice preferences, and refine rubrics, which in turn steer data curation ( 2.1 ) and objective shaping ( 2.2 ). In this sense, evaluation becomes a form of post-hoc guidance rather than passive measurement. The result is an iterative alignment cycle that balances raw performance with usability, trust, and safety.

 
 

#### 2.3.1. Trust, Transparency, and Safety 

 
 Establishing calibrated trust begins with making model limits visible. Bansal et al. ( bansal2019 ) show that users build more accurate mental models when an AI system discloses its own error likelihood. Zhou and Chen’s Uncertainty - Performance Interface (UPI) ( Jianlong2019 ) couples confidence scores with outcome heat - maps so that decision makers can dynamically adjust their thresholds. Recent work extends these ideas to multimodal settings: Ji et al. ( align-anything-200k ) let raters score image–text consistency, while Vodrahalli et al. ( vodrahalli2025canonical ) distill 21 canonical preference axes that cover most harm - related judgements, trimming annotation cost without losing coverage. Together, these studies frame transparency as an ongoing dialogue rather than a static disclosure.

 
 
 

#### 2.3.2. Collaborative Approaches to Improve Evaluation 

 
 Human contribution is most powerful when it shapes evaluators, not just evaluations. Clark et al. ( clark2021 ) boost label accuracy by providing annotators with contrasting model outputs and rationale. They examine how human evaluators can differentiate between texts produced by humans and models like GPT-2 and GPT-3. They suggest methods to enhance evaluation accuracy, including detailed instructions, annotated examples, and text comparisons (Fig.  6 ). Chhibber et al. ( Chhibber2022 ) demonstrate that teaching annotators the model’s decision heuristics increases both trust and delegation willingness. InstructRL ( hu2023 ) and Tabrez et al. ( Tabrez2020 ) further show that natural - language feedback can be folded back into the policy itself, reducing the evaluation–training gap. Error - analysis prompting (EAPrompt) with chain-of-thought reasoning ( lu2023 ) and collaborative detection workflows ( uchendu2023 ) route annotator attention to likely failure modes, lowering variance and cost.

 

 
 
 Overall, the work on human evaluation shows a clear move away from static performance metrics toward an ongoing, people-centered checking of model behavior. As LFM outputs become more fluent and convincing, automated benchmarks alone cannot capture qualities like trust, safety, or transparency. This puts humans in a different position, not just as scorers, but as active partners who stress-test the system and surface the kinds of failures that models are best at hiding. The hard part is not only collecting those judgments at scale, but figuring out how to train and support evaluators so they can apply consistent rubrics, recognize subtle flaws, and probe these systems in ways that automated tests simply cannot. Together with data curation and objective shaping, human evaluation closes the build-time loop of human-guided model development: humans influence what the model sees, what it optimizes, and how its behavior is judged. This differs from the run-time collaboration patterns discussed in Sec. 3 , where the model is already deployed and the central question becomes how humans and AI divide initiative, responsibility, and control during task execution.

 
 
 
 
 

## 3. Design Principles for Human-AI Collaboration: Working with the AI Partner 

 
 We now turn to the question of how humans and AI systems collaborate after deployment. At this run-time stage, the model is no longer only an object to be trained or evaluated, but a partner whose interface, role, initiative, and control boundaries must be carefully designed.

 
 
 Effective human-AI teams rely on more than raw model accuracy; they require systems intentionally designed for collaboration. This section lays out core design principles that turn powerful large pre-trained foundation models into reliable partners: (i) crafting interaction channels that surface the right information at the right moment, (ii) allocating initiative so humans and AI can fluently guide or assist one another, and (iii) embedding continuous evaluation and adaptation to keep the partnership usable, trustworthy, and resilient as tasks, data, and user expectations evolve.

 
 

### 3.1. Interaction and Interface Design

 
 Before roles and control can be allocated, human-AI collaboration requires interaction channels through which people can understand, guide, correct, and trust the AI system. Interfaces, therefore, act as the practical surface where run-time collaboration becomes possible: they expose model behavior, support user input, and make the system’s state and uncertainty legible.

 
 
 Effective interactions and interfaces translate AI reasoning into human-understandable cues, enabling seamless and timely communication between humans and AI. To support human-AI collaboration effectively, it is crucial to focus on personalizing interactions, facilitating natural communication channels, and providing transparent explanations of AI behavior.

 
 

#### 3.1.1. Personalization and Adaptive Displays 

 
 Integrating AI agents into collaborative teams has historically been challenging due to limited adaptability and personalization. Early AI systems, designed for narrowly defined tasks, struggled to grasp human subtleties or adjust dynamically to evolving team requirements, hindering effective collaboration ( zhang2021 ) . The advent of LLMs significantly enhanced these capabilities. Leveraging extensive training datasets and sophisticated algorithms, LLM-equipped AI agents now exhibit improved adaptability, dynamically adjusting autonomy levels to match team workflows and individual user preferences ( Hauptman2023 ) . Additionally, LLMs facilitate richer conversational interactions, providing context-sensitive, multi-turn responses and enabling AI to mediate complex interactions among human team members ( Bryan2023 ; Qing2023 ) . This enhanced personalization extends to mobile UI/UX evaluations, where LLMs analyze usability test videos to tailor user interactions effectively ( Mingming2023 ) . Techniques such as chaining LLM prompts further enhance transparency and user control, customizing collaboration experiences according to user needs ( Tongshuang2022 ) . This way, personalization is not only a usability feature, but also a mechanism for role calibration: the interface adapts what the AI shows, asks, or takes over based on the user’s goals, expertise, and context.

 
 
 

#### 3.1.2. Conversational and Multimodal Interaction 

 
 Effective human-AI collaboration relies heavily on clear and natural communication, influencing trust, information exchange, and team coordination ( Kozlowski2006 ; Salas2017 ) . Advances in AI conversational capabilities now enable responsive and contextually appropriate dialogue, bolstering user confidence and improving team effectiveness ( Zhang2023 ) . Notably, teachable conversational agents exemplify significant progress, learning dynamically from conversational interactions and adapting their knowledge to meet evolving task demands ( Chhibber2022 ) . These agents’ perceived likability and human-likeness directly enhance their communication efficacy ( Clark2016 ) . Furthermore, natural language processing (NLP) continues to refine voice-based interactions, enhancing both verbal and textual exchanges within collaborative teams ( wang2019human ) .

 
 
 Efforts have also been devoted to establishing standardized communication frameworks within HAI collaboration teams. Initiatives like developing a shared vocabulary (Taxonomy Model) and comprehensive communication and explanation models facilitate clear, mutual understanding of AI actions and rationale, significantly boosting collaborative efficiency and team cohesion ( Bosch2019 ) . Conversational and multimodal interfaces, thus, provide a communication surface for mixed-initiative collaboration, where either the human or the AI can propose, clarify, revise, or redirect the task.

 
 
 

#### 3.1.3. Transparency, Feedback and Explainability 

 
 The principles of user interface design (UI) and user experience design (UX) have progressively emphasized transparency, ethical interaction, and clear feedback mechanisms in collaborative human-AI systems. Early frameworks emphasized usability testing, clear communication, and synchronization between human and AI team members, laying foundations for trustworthy interactions ( Mingming2023 ; Flathmann2021 ) . Recent innovations leveraging large-scale pre-trained models have transformed these traditional UI/UX designs. For example, two-stage frameworks like ChatIE reframe zero-shot Information Extraction (IE) as interactive, multi-turn question-answering sessions, achieving superior performance and enabling transparent user interactions ( wei2023 ) . Empirical studies involving diverse user groups confirm these advances significantly enhance both interaction quality and emotional satisfaction, reinforcing AI’s ability to adopt human-like behaviors and integrate seamlessly into complex social contexts ( Lemaignan2017 ) . Transparency and explainability form the control layer of human-AI interaction: they help users inspect the AI’s reasoning, detect failure modes, and decide when to accept, revise, or override its output.

 

 
 
 This reveals a central tension in LFM-based collaboration. We are moving from rigid graphical interfaces to fluid, conversational interactions, which makes AI systems easier to approach but can also weaken precision, constraint enforcement, and user control. Natural language can hide system state and reasoning, while structured interfaces make parameters, validation, and recovery actions more explicit. The open challenge is therefore to design hybrid interfaces that combine natural-language flexibility with visible reasoning, structured controls, previews, confirmations, and undo mechanisms, so users retain clarity and agency as collaborative tasks become more complex.

 
 
 Figure 7 . Collaboration patterns and role allocation in HAI systems. Section 3.2 organizes run-time collaboration into human-steered AI, AI-augmented human work, mixed-initiative interaction, control handoff and delegation, and emerging supervisory orchestration of agentic systems. The dashed boundary highlights supervisory orchestration as a newer mode where humans monitor, guide, and intervene in multi-step AI workflows. Five panels illustrate recurring Human-AI collaboration patterns. Five panels are arranged from left to right under the heading Collaboration Patterns and Role Allocation. Human Steers AI shows a person directing a robot. AI Augments Human shows a robot guiding or assisting a person. Mixed Initiatives shows a person and robot shaking hands. Control Handoff and Delegation shows them transferring a tool. Supervisory Agentic Orchestration shows a person monitoring a robot performing a task. The final panel has a dashed boundary to identify supervisory orchestration as an emerging collaboration pattern. 
 
 
 
 

### 3.2. Collaboration Patterns and Role Allocation

 
 The interaction channels described above become useful only when they are tied to clear collaboration patterns: who initiates action, who provides support, who makes final decisions, and when control shifts between partners. We organize this design space into recurring patterns that appear across LFM-based HAI systems: Human steers AI, AI Augments Human, Mixed-Initiative Collaboration, Dynamic Delegation/Control Handoff, and the emerging pattern of Supervisory Orchestration in agentic systems (Fig. 7 ). They provide a practical vocabulary for connecting interface choices to role allocation, control, failure modes, and safeguards (Table 2 ). Effective human-AI collaboration relies on clearly defining interaction patterns and thoughtfully allocating roles to maximize the strengths of both partners.

 
 

#### 3.2.1. Human steers AI 

 
 This pattern captures cases where humans actively steer the AI during use, through prompting, in-context correction, and contextual oversight by adjusting its behavior and level of autonomy as the task unfolds, rather than only shaping it before deployment (Sec. 2 ). Human expertise plays an essential role in enhancing AI capabilities throughout various development stages, from data preparation and algorithm refinement to ethical and contextual oversight ( wei2023 ; Tongshuang2022 ; Pflanzer2022 ) . Accurate data annotation and tailored algorithm adjustments guided by human feedback significantly improve AI precision and applicability ( dubey2020 ; liu2023tag ; ngo2023tag ) . Humans also ensure AI adherence to societal norms and ethical standards, aspects that purely algorithmic approaches might overlook ( Pflanzer2022 ) .

 
 
 Further, intuitive interface designs facilitate effective human-AI communication, ensuring that sophisticated AI outputs remain accessible and practically useful ( Zhang2023 ) . Continuous human input and real-world behavioral data enhance specialized AI systems, such as adaptive driving systems, integrating subtle human behaviors into AI decision-making ( Gopinath2022 ; Munyaka2023 ) . At run time, this steering is expressed largely through natural-language instruction and preference signals, the deployment-side counterpart to the training-time preference methods covered in Sec. 2.2 (e.g., RLHF, DPO, InstructRL ( rafailov2023 ; hu2023 ) ), letting users align AI outputs with their expectations without retraining the model.

 
 
 Table 2. Collaboration patterns for HAI LFM systems. The table summarizes recurring ways in which humans and AI share roles, initiative, control, and responsibility, along with typical use cases, risks, and safeguards. 
 
 
 
 
 
 Pattern 
 | 
 
 
 Human role 
 | 
 
 
 AI role 
 | 
 
 
 Good for 
 | 
 
 
 Risks 
 | 
 
 
 Safeguard 
 | 

 
 
 
 Human-Steers-AI 
 | 
 
 
 Steers AI through instructions, corrections, preferences, or evaluation 
 | 
 
 
 Incorporates human input and adjusts its behavior accordingly 
 | 
 
 
 Run-time steering, adaptation, alignment, and auditing 
 | 
 
 
 Feedback may be biased, inconsistent, or too narrow 
 | 
 
 
 Use diverse feedback, train evaluators, define clear rubrics, and include adversarial checks 
 | 

 
 
 
 AI-Augments-Human 
 | 
 
 
 Makes the final decision and uses AI as support 
 | 
 
 
 Suggests, summarizes, explains, drafts, or retrieves useful information 
 | 
 
 
 Healthcare, education, accessibility, and expert decision support 
 | 
 
 
 Users may over-trust the AI or miss its uncertainty 
 | 
 
 
 Show confidence, explain outputs, allow override, and calibrate trust 
 | 

 
 
 
 Mixed-Initiative Collaboration 
 | 
 
 
 Works with the AI as a co-designer, reviewer, or task partner 
 | 
 
 
 Suggests ideas, critiques outputs, revises work, or redirects the task 
 | 
 
 
 Creative work, testing, counseling, games, and design/code workflows 
 | 
 
 
 Roles, authorship, or next steps may become unclear 
 | 
 
 
 Use clear turn-taking, visible reasoning, revision history, and confirmation steps 
 | 

 
 
 
 Dynamic Delegation / Control Handoff 
 | 
 
 
 Steps in as the expert, driver, analyst, or supervisor when needed 
 | 
 
 
 Flags uncertainty, triages cases, escalates risk, or transfers control 
 | 
 
 
 Healthcare deferral, autonomous vehicles, and security alerts 
 | 
 
 
 Handoffs may happen too late, too often, or when the human is unprepared 
 | 
 
 
 Set escalation rules, monitor human readiness, and maintain shared situational awareness 
 | 

 
 
 
 Emerging Supervisory Orchestration 
 | 
 
 
 Sets goals, reviews plans, monitors progress, and intervenes when needed 
 | 
 
 
 Plans, uses tools, executes multi-step tasks, or coordinates agents 
 | 
 
 
 Agentic workflows, tool-use systems, and long-horizon tasks 
 | 
 
 
 Errors may accumulate silently; tool use and responsibility may become opaque 
 | 
 
 
 Require plan review, approval checkpoints, execution traces, interruptibility, and audit logs 
 | 

 

 
 
 

#### 3.2.2. AI Augments Human 

 
 AI systems increasingly serve as critical collaborators, enhancing human capabilities by streamlining complex tasks and decision-making processes. Traditional AI applications, such as diagnostic aids and autonomous vehicle controls, have evolved significantly, promoting transparency and trust in collaborative settings ( wei2023 ; Jianlong2019 ) . In healthcare, AI-driven tools like TREWS offer clinicians timely support, effectively managing large datasets for critical interventions ( henry2022 ) . Similarly, advanced human-machine collaboration tools, such as intelligent haptic interfaces, improve task safety and efficiency, particularly during critical transitions like automated to manual vehicle control ( Lv2021 ) .

 
 
 Recent advancements in LLMs have further enhanced collaborative dynamics, supporting complex task management through chaining prompts, offering transparency and control in interactions ( Tongshuang2022 ) . LLMs augment human productivity and creativity, aiding tasks from brainstorming to detailed architectural design, highlighting AI’s role in enhancing rather than replacing human input ( wang2023 ; Veselovsky2023 ; pal2023 ; Memmert2023 ; ahmad2023towards ) . Additionally, AI-driven delegation managers demonstrate effective dynamic control allocation, enhancing operational safety and collaboration efficiency in human-AI systems ( Fuchs2023 ) .

 
 
 

#### 3.2.3. Mixed-Initiative and Complementary Strengths 

 
 This pattern describes settings where both humans and AI can propose, critique, revise, or redirect the task, rather than assigning initiative to only one side. Combining human cognitive skills with AI’s computational power often yields outcomes superior to individual capabilities. Effective collaboration involves understanding each other’s strengths and limitations, minimizing overlapping errors, and maximizing mutual correction ( bansal2021does ; Chakraborti2017 ) . Initially, mismatches in human understanding of AI capabilities can cause inefficiencies; however, over time, humans adapt their mental models, leading to enhanced collaboration ( Bosch2019 ; liu2023humans ) . Hybrid AI approaches, such as neuro-symbolic frameworks, now actively predict and adapt to human behaviors, improving cooperation through sophisticated psychological and emotional modeling ( Bosch2019 ; Neerincx2018 ; Nikolaidis2017 ; Zhao2022 ; Kaptein2016 ) .

 
 
 Practical applications demonstrate this synergy, such as AdaTest++ which collaboratively audits LLM reliability, combining human intuition with AI analysis ( Rastogi2023 ) . The HAILEY system, supporting mental health conversations, exemplifies AI augmenting human empathy with analytical depth ( Sharma2023 ) . Furthermore, creative integrations, like combining LLMs with diffusion models for visual metaphor creation, illustrate the potential of collaborative interactions to bridge conceptual creativity and tangible outputs ( Chakrabarty2023 ) .

 
 
 

#### 3.2.4. Control-Handoff and Delegation 

 
 Effective human-AI teaming necessitates fluent control handoffs and clearly defined delegation protocols. Dynamic allocation of control between humans and AI, based on situational awareness and respective strengths, optimizes team performance and safety ( Sharifi2022 ) . Research highlights the importance of smooth transitions, particularly in safety-critical environments like driving systems, where intelligent delegation managers allocate tasks intelligently based on real-time assessments ( Fuchs2023 ) . This adaptive control handoff ensures responsiveness to rapidly changing conditions, fostering a robust, resilient, and effective collaborative system.

 
 
 

#### 3.2.5. Emerging Supervisory Orchestration in Agentic Systems 

 
 A newer collaboration pattern is emerging as LFMs are increasingly organized into agentic and multi-agent systems, where AI agents can act over multi-step workflows rather than only produce a single response. In these systems, one or more AI agents do not simply answer a single prompt; they can plan steps, call tools, divide subtasks, critique intermediate outputs, and revise their behavior based on feedback. Recent work shows this shift in several directions: multi-agent planning and orchestration for complex tasks ( fourney2024magenticone ) , self-refinement systems where different agents take on generator, critic, verifier, or refiner roles ( motwani2024malt ; yu2025table ; vats2026guideline ) , and human-centered multi-agent interfaces that move from direct conversation to orchestration, raising design questions around how users supervise coordinated agents, resolve conflicts, and retain control without micromanaging every step ( schoembs2026conversation ; masters2025orchestrating ) .

 
 
 These systems introduce the pattern we call supervisory orchestration. Here, the human is no longer only accepting or rejecting a model output. Instead, the human defines the goal, sets constraints, reviews plans or execution traces, approves high-impact actions, and intervenes when the agent drifts from the intended task. This is closer to human- on -the-loop supervision than traditional human-in-the-loop annotation, because the human does not guide every step but remains responsible for monitoring and redirecting the overall process.

 
 
 Recent benchmarks make this pattern especially important because they evaluate agents in longer, more realistic workflows. τ \tau -bench tests agents in dynamic user conversations where both tool use and domain-specific policy following must be correct ( yao2024tau ) . OSWorld evaluates multimodal computer-use agents in realistic desktop environments ( xie2024osworld ) , while OSWorld-Human shows that even successful agents may take far more steps than humans, making efficiency and user burden part of collaboration quality ( abhyankar2025osworld ) . TheAgentCompany extends this question to workplace-like tasks involving browsing, coding, file use, and communication ( xu2026theagentcompany ) . In healthcare, MedAgentBench evaluates agents inside a virtual electronic health record environment, showing how agentic workflows may enter high-stakes professional settings ( jiang2025medagentbench ) . Together, these studies suggest that agentic systems should not be evaluated only by final task success. A human supervisor also needs to know whether the plan was reasonable, whether tool calls were appropriate, whether policies were followed, and whether the agent created risk while reaching its final answer.

 
 
 Thus, supervisory orchestration extends the earlier patterns in this section. It builds on mixed-initiative collaboration because both humans and AI can revise the task, and it builds on dynamic delegation because control may shift depending on risk, uncertainty, or task progress. However, it also introduces a distinct role-allocation problem: the human is no longer only supervising an output, but supervising a process. This connects directly to the system-level evaluation concerns in Sec. 3.3 and the broader open challenges discussed in Sec. 6 .

 

 
 
 This collaboration research shows a constant balancing act: letting the AI handle more of the heavy lifting without sidelining the human. LFMs can now step in as proactive partners, but that often leaves people in a narrow role of supervisor or checker, risking skill loss and disengagement. The real challenge is that there’s no single right division of labor. The appropriate pattern depends on the task, the user’s expertise, the model’s confidence, and the risk of failure. Future HAI systems, therefore, need adaptive role-management mechanisms that can hand off control smoothly, so humans stay engaged and in the loop while still benefiting from the AI’s efficiency.

 
 
 
 

### 3.3. System Integration and Evaluation

 
 Once collaboration patterns are defined, they must be embedded into real workflows and evaluated as complete human-AI systems. This requires more than inserting an LFM into an existing pipeline: the system must preserve useful human practices, support users as they learn new roles, measure collaboration quality beyond raw task performance, and adapt as models, users, and organizational needs change.

 
 
 Integrating human-AI systems within real-world workflows demands both seamless technical alignment and rigorous assessment across multiple dimensions. This section examines four pillars critical to sustainable collaboration: ensuring compatibility with existing infrastructures and organizational processes, supporting users through intuitive training and interfaces, measuring collaborative performance holistically, and maintaining system efficacy through continuous monitoring and adaptation.

 
 

#### 3.3.1. Backwards Compatibility 

 
 In the current fast-growing AI, ensuring backward compatibility is essential for integrating new technologies into existing systems without disruption. This is especially important when bringing AI into legacy infrastructures, where long-established tools, processes, and workflows must be considered. Weisz et al. ( Weisz2021 ) explore how generative models can support application modernization, highlighting the importance of designing AI systems that can work seamlessly with older architectures. Similarly, Chhibber et al. ( Chhibber2022 ) emphasize the value of aligning AI tools with established workflows in traditional crowd work, showing how models like GANs and autoencoders can enrich and improve datasets within existing setups. Bau et al. ( Bau2019 ) also present a creative use of GANs to tailor image priors to the unique characteristics of individual images, addressing challenges in high-level semantic editing tasks.

 
 
 Beyond technical integration, compatibility challenges also arise at the societal and economic levels. Wang ( wang2023 ) discusses how LLMs are reshaping the job market, stressing the need to introduce these technologies thoughtfully to avoid major disruptions. Eloundou et al. ( Eloundou2023 ) further examine this shift, estimating that around 80% of U.S. workers may see some changes in their tasks due to LLMs, with 19% potentially experiencing over half of their tasks transformed. Finally, Harrer et al. ( harrer2023 ) focus on the responsible use of LLMs in generative applications, particularly in fields like healthcare. They argue for strong human oversight and ethical design to prevent misuse, underscoring the potential of LLMs to be both effective and trustworthy when deployed responsibly.

 
 
 

#### 3.3.2. Learning Curve, Training and Usability 

 
 McNeese et al. ( McNeese2021 ) show that when AI takes a leadership role in a team, overall performance can improve but human partners often face an initial adjustment period. Lyons et al. ( lyons2019 ) argue that building interfaces that feel like true collaborators, rather than mere tools, is key to shortening that learning curve.

 
 
 A foundation pillar of usable HAI is explainability. Hou et al. ( Hou2023 ) demonstrate that users trust AI more when they see why it made a particular choice, even if that choice seems unexpected. Bansal et al. ( bansal2021does ) reinforce this: clear AI explanations directly boost human confidence in the system.

 
 
 Looking at LLMs, Wu et al. ( Tongshuang2022 ) find that breaking complex tasks into a chain of smaller prompts helps users guide the model more effectively, calibrate its responses, and validate each sub-task. Veselovsky et al. ( Veselovsky2023 ) caution, however, that while LLMs often generate polished, consistent answers, they may miss the spontaneity of genuine human input. Their study reveals that workers using LLMs still produce higher-value outputs than those without, underscoring the importance of ongoing monitoring and adaptation as humans and AI evolve together. Training and usability, thus, support role calibration: users must learn not only how to operate the AI system, but also when to rely on it, when to question it, and when to override it.

 
 
 

#### 3.3.3. Collaboration Metrics: Task, Team, User Experience 

 
 Measuring how well humans and AI work together means looking beyond raw accuracy to three key areas: the task itself, the team dynamics, and the user’s experience. Henry et al. ( henry2022 ) stress that HAI systems must earn user trust and support autonomy; evaluations should check whether domain experts find the system intuitive and reliable. Liu et al. ( liu2023humans ) add that we need to capture how human and machine perceptions differ, and then design metrics that show how their complementary strengths improve joint decisions. Munyaka et al. ( Munyaka2023 ) dive into social dynamics, how revealing an AI’s identity or different decision styles affects team cohesion and effectiveness. In creative settings, Memmert et al. ( Memmert2023 ) remind us to assess not just AI’s cognitive contributions but also how it influences group behaviors like participation and free-riding. Finally, Chang et al. ( chang2023 ) offer a broad framework for LLMs that combines task performance, reasoning, robustness, and ethical considerations, underscoring that true HAI success is judged by a blend of technical, social, and experiential measures.

 
 
 

#### 3.3.4. Continuous Adaptation and Post-Deployment Monitoring 

 
 Even a well-tuned HAI system can drift if human workflows or real-world conditions change. Veselovsky et al. ( Veselovsky2023 ) demonstrate the value of observing crowd workers as they interact with LLMs and using those insights to iteratively refine both prompts and model behavior. Chang et al. ( chang2023 ) also highlight the importance of tracking bias, fairness, and ethical impacts after deployment. A robust monitoring plan involves logging interactions, surveying users for unexpected behaviors, and rolling out incremental updates that preserve backward compatibility. By continuously collecting feedback, both quantitative (e.g., error rates, response times) and qualitative (e.g., user satisfaction, trust), teams can adapt models, interfaces, and collaboration protocols to keep HAI partnerships effective and aligned with real-world needs. Post-deployment monitoring is especially important for LFM-based systems because failure can emerge not only from model drift, but also from workflow drift: users may over-rely on the system, ignore uncertainty signals, or adapt their behavior in ways the original evaluation did not capture.

 

 
 
 System integration efforts indicate one thing: there is a mismatch between the speed of LFM innovation and the slower, more cautious pace of deploying them in high-stakes workflows. Traditional metrics like accuracy or speed don’t capture what really matters in collaboration. Generative partners may complete tasks quickly, but that tells us little about user trust, mental load, or how well the human-AI team actually works together. The open challenge is building evaluation frameworks that look beyond one-off performance. We need ways to measure long-term factors, user experience, team dynamics, adaptability so we can judge not just if the system works, but if the partnership is resilient and effective over time.

 
 
 
 
 

## 4. Ethical, Societal and Governance Aspects of Human-AI Collaboration

 
 The collaboration patterns discussed above do not operate in isolation. Whether humans steer models, receive AI assistance, share initiative, or hand off control, each mode raises cross-cutting questions about fairness, autonomy, privacy, accountability, and policy. These concerns determine whether a human-AI partnership is not only technically effective, but also socially acceptable and responsible. In this section, we examine how human-AI collaborative teams can be designed and governed to ensure fairness, support autonomy and well-being, manage labor impacts, safeguard data, foster trust, and guide policy.

 
 
 Figure 8 . Large Language Models often exhibit gender bias due to primarily being trained on large datasets of human language that reflect existing societal biases. Kotek et al. ( kotek2023 ) test this by probing LLMs to answer permutations of complex questions, as shown above. The answers and their explanation help the authors to quantify the bias. A two-by-two prompt design varies occupational roles, sentence order, and pronoun gender to test bias. Four prompts use the occupations executive and secretary, traditionally associated with male and female stereotypes, respectively. The prompts systematically reverse the order of the occupations and vary the pronoun between she and he. Each sentence asks whether the executive or secretary had to read a memo. Comparing model answers across these four permutations reveals whether pronoun resolution is influenced by occupational gender stereotypes. 
 
 

### 4.1. Fairness in Collaborative Decision-Making

 
 Algorithmic fairness remains challenging as models inherit biases from data and design choices ( Ben2019 ; Stahl2021 ) . Ensuring representative datasets, transparent logic, and accountability is fundamental ( Nicholas2023 ) . In mixed - initiative workflows, human oversight helps surface unfair patterns early: Chhibber et al. ( Chhibber2022 ) show interactive feedback loops refine model behavior, and Morrison et al. ( Morrison2023 ) highlight how causal explanations empower users to correct errors. Studies on LLMs reveal persistent stereotypes: Kirk et al. ( kirk2021 ) document occupational biases in GPT - 2, Huang et al. ( huang2023bias ) identify sensitive - attribute bias in generated code, and Kotek et al. ( kotek2023 ) quantify amplified gender stereotypes (Fig. 8 ). Meyer et al. ( meyer2023chatgpt ) discuss ChatGPT’s writing-assist strengths alongside its bias and factuality pitfalls. Recent mitigation surveys outline stage - wise strategies: Li et al. ( li2023bias ) review size - specific debiasing, Gallegos et al. ( gallegos2023bias ) propose bias - metric taxonomies, Ohi et al. ( ohi2024bias ) leverage few - shot corrections, and Bi et al. ( bi2023bias ) introduce chain - of - thought methods. In resume screening, balanced human–AI teams improve perceived fairness and trust ( ling2024applicants ) .

 

 
 
 Fairness in human-AI collaboration faces a paradox: human oversight is meant to reduce bias, but often reinforces it through automation and confirmation bias. Since LFMs inherit stereotypes from vast internet data, simply adding a human is not enough. The key challenge is building fair-by-design frameworks that let humans question, audit, and correct model outputs, acting as true reviewers rather than rubber stamps.

 
 
 

### 4.2. Empowering Human Autonomy and Well-Being

 
 Chhibber et al. ( Chhibber2022 ) demonstrate how teachable conversational agents in crowd-based applications can empower workers, enhancing control, personalized learning, and ownership, thereby boosting job satisfaction. Konstantis et al. ( Konstantis2023 ) emphasize transparency and fairness in crowdsourcing platforms, advocating for clear decision-making and respectful treatment to improve worker experience. Furthermore, Pal ( pal2023 ) highlights the design of LLMs focused on worker well-being and autonomy, avoiding stress-inducing features and promoting engagement and learning.

 
 
 Despotovic and Bogodistov ( despotovic2024 ) highlight a trend in job seekers preferring roles that incorporate advanced AI, like ChatGPT, aligning with their identities and technological interests. This shift towards AI-interactive jobs indicates a broader preference for technologically engaging roles, impacting job satisfaction and autonomy. Complementarily, Wang ( wang2023 ) explores the paradox of LLMs in the job market, noting their role in both creating new opportunities and obsoleting certain jobs. This dual effect is key in assessing worker autonomy and well-being, as LLM-driven automation fosters more creative roles and autonomy but also raises concerns about job security and satisfaction. Integrating insights from recent research, it is evident that while generative AI has the potential to automate menial tasks and foster more creative roles, it also poses challenges to worker autonomy, necessitating a balanced approach to AI deployment in the workplace ( woodruff2023 ) .

 

 
 
 As LFMs get more capable, the risk is that humans become passive consumers, losing skills, engagement, and ownership. The challenge is not maximizing what the AI can do, but creating systems that support human growth. Future designs should treat AI as a platform for learning and creativity, not a replacement for effort, so collaboration strengthens autonomy instead of eroding it.

 
 
 

### 4.3. Collaborative Labor Dynamics

 
 AI redefines work by automating routine tasks, changing existing roles, and creating new forms of human-AI collaboration. Prior work suggests that AI can improve productivity and business performance when integrated into organizational workflows ( Purdy2016 ; CHOWDHURY202231 ) , while delegation to AI can also improve human task performance and satisfaction in some settings ( Hemmer2023 ) . At the same time, labor-market analyses suggest that LLMs may expose many occupations to task restructuring, displacement pressure, and reskilling needs ( Eloundou2023 ; wang2023 ; walkowiak2023 ) . Chhibber et al. ( Chhibber2022 ) observe that crowdworkers can upskill when supervising teachable agents, and Pal ( pal2023 ) anticipates growth in AI-centered roles. Job seekers increasingly blend AI into their professional identities ( despotovic2024 ) , highlighting the need for reskilling and value-sharing models. 
 

 
 
 Generative AI promises big productivity gains but also raises the risk of job loss and de-skilling. While the ideal is augmentation, freeing people for more creative work, the reality is that many roles will be reshaped or eliminated. The real question is how to manage that shift fairly. Future work should focus on reskilling pipelines and new economic models, so the benefits of human-AI collaboration are shared broadly, not concentrated in a few hands.

 
 
 

### 4.4. Data Privacy and Security

 
 Data privacy and security are the foundation of any human-AI partnership, and nothing undermines it faster than opaque or insecure data practices. Ezer et al. ( Ezer2019 ) introduce the idea of dynamic trust engineering, where system transparency and user controls adapt in real time to changing context and risk. Building on this, Yin et al. ( Yin2019 ) show how combining logistic regression with differential privacy can give analysts strong guarantees of individual anonymity while still delivering accurate insights.

 
 
 In sensitive domains such as healthcare, federated learning offers a path forward: Kaissis et al. ( Kaissis2020 ) demonstrate how models can improve across hospitals without ever exposing raw patient data. Human supervisors can review each local update before it’s aggregated, catching anomalies early. Likewise, Samyuktha et al. ( Admin2022 ) survey AI-driven anomaly detection to spot suspicious access or data exfiltration, reinforcing cyber defenses in mixed-initiative workflows. Lepri et al. ( Lepri2021 ) argue that a human-centric privacy-by-design ethos where user consent, minimal data collection, and clear accountability are baked into every feature builds long-term confidence in AI systems.

 
 
 Large Generative AI Models (LGAIMs) like ChatGPT and GPT-4 bring fresh privacy and security challenges. Hacker et al. ( Hacker2023 ) propose a regulatory framework that mandates clear documentation of data sources, risk assessments for sensitive applications, and ongoing transparency reports. To address privacy at the model level, Ullah et al. ( ullah2023 ) introduce PrivChatGPT, which embeds differential privacy directly into LLM training. Gupta et al. ( gupta2023 ) highlight GenAI’s new attack vectors, prompt injection and data poisoning, and recommend integrating ethical guidelines with robust cybersecurity measures.

 
 
 Meanwhile, De Angelis et al. ( de2023 ) warn that unfettered LLM outputs can fuel misinformation ’infodemics’, calling for policy interventions and automated fact checking layers. Thapa and Adhikari ( thapa2023 ) stress strict validation pipelines for biomedical AI to prevent diagnostic errors and data leaks. Finally, Sebastian ( sebastian2023 ) emphasizes data minimization, retaining only task-essential features, and pairing it with federated or decentralized architectures to further reduce exposure. When these technical safeguards sit alongside clear user controls and transparent audit logs, human-AI teams can share data confidently, knowing privacy and security remain front and center.

 

 
 
 Bringing LFMs into collaborative workflows heightens the trade-off between personalization and privacy. Rich, contextual data makes interactions smoother, but it also exposes users to risks like prompt injection or data leaks. The challenge is to design privacy-preserving architectures through methods like federated learning, on-device execution, or differential privacy, so people can collaborate with AI confidently without giving up control of their data.

 
 
 

### 4.5. Building Trust and Shared Accountability

 
 Building trust in human–AI collaborative teams starts with clear ethical principles, human oversight, and system transparency. Hou et al. ( Hou2023 ) stress secure architectures that surface potential risks while ensuring humans can intervene at any point. Pflanzer et al. ( Pflanzer2022 ) extend this by framing decisions in the Agent-Deed-Consequence model, which logs each actor’s intent, action, and outcome, creating an auditable trail. In cooperative settings like multiplayer gaming, Caldwell et al. ( caldwell2022agile ) show that social factors, peer behaviors and shared norms, are just as important as technical safeguards for fostering trust.

 
 
 Large language models introduce new dimensions to this landscape. In healthcare and other high-stakes settings, trust should not mean passive acceptance of fluent model outputs but should be calibrated through transparency, expert validation, and continued human oversight. Choudhury and Chaudhry ( choudhury2024large ) emphasize that clinicians must balance trust and skepticism when using LLMs, since over-reliance can weaken professional judgment and contribute to deskilling. Kim et al. ( kim2023help ) take this further by embedding trial-by-trial uncertainty into each AI suggestion, so users see not only what the model proposes but how confident it is. Finally, Bansal et al. ( bansal2019 ) reveal that exposing error bounds helps users build accurate mental models of AI behavior, which is vital for effective joint decision-making.

 
 
 

### 4.6. Policy and Regulation for Human-AI Collaborative Teams

 
 Building reliable human-AI partnerships requires clear, enforceable rules. Early surveys of ethics guidelines show a global consensus on transparency, accountability, fairness and bias mitigation ( jobin2019 ; cath2018 ) . Stahl et al. ( Stahl2021 ) distill these into six pillars; Transparency, Accountability, Bias Mitigation, Human Oversight, Data Protection and Public Education, that remain the cornerstone of policy for any collaborative AI system.

 
 
 The rise of large language models has exposed gaps in existing laws. Bender et al. ( Bender2021 ) warn that unchecked data harvesting and embedded biases in LLMs demand new statutes around training data origin. Recent legal analyses ( mit2023law ) of cases such as the use of open source code by GitHub Copilot illustrate how copyright and attribution rules must evolve to protect both creators and users. Ekenobi et al. ( Chinonso2023 ) praise ChatGPT’s opt-in privacy features, but call for uniform data protection standards across platforms. Meanwhile, Marcos and Pullin ( gdpr2023law ) highlight the EU’s ongoing struggle to balance innovation with individual rights under GDPR; an example of how regulators must adapt swiftly to keep human-AI teams both compliant and cutting-edge.

 

 
 
 Collectively, the work on trust, accountability, and policy brings out a critical governance gap. The speed of LFM development has far outpaced the creation of clear legal and regulatory frameworks, creating a disparity between fostering innovation and ensuring public safety. Simply appealing to high-level ethical principles is insufficient. When a collaborative human-AI team makes a harmful decision, the lines of responsibility are blurred, making accountability nearly impossible to establish. The most significant challenge ahead is to translate abstract principles into concrete, auditable, and enforceable standards. This requires a multi-stakeholder effort to create clear documentation requirements, auditable decision trails for HAI systems, and regulatory safe harbors that encourage responsible innovation while establishing clear liability for when things go wrong.

 
 
 
 

## 5. Applications: Collaboration Across Domains 

 
 The preceding sections described how humans shape LFMs during development, how human-AI roles are allocated at run time, and how ethical and governance constraints shape these partnerships. This section examines how those collaboration patterns appear in concrete domains. Healthcare emphasizes calibrated AI augmentation and expert deferral; autonomous vehicles foreground dynamic control handoff; surveillance and security rely on triage and supervisory oversight; games highlight mixed-initiative collaboration; education centers teacher-guided augmentation; and accessibility requires co-adaptive personalization. Across these domains, the central question is not whether LFMs can automate a task, but how human judgment, control, and accountability are preserved when these models enter real workflows.

 
 

### 5.1. Healthcare

 
 Human-AI teams are transforming patient diagnosis, treatment planning, and clinical documentation. Henry et al. ( henry2022 ) , Bienefeld et al. ( Bienefeld2023 ) and Memmert et al. ( Memmert2022 ) demonstrate that integrating clinician feedback into AI predictions boosts diagnostic accuracy, while Carrie et al. ( Carrie2019 ) show how AI-driven image retrieval accelerates case review. AI systems have even matched expert performance in breast cancer screening ( McKinney2020 ) , but clinical adoption hinges on trust: Budd et al. ( BUDD2021 ) and Choudhury et al. ( CHOUDHURY2022 ) find that transparent explanations and hands-on training are critical for user acceptance.

 
 
 Large FMs bring fresh capabilities. Lyu et al. ( lyu2023 ) illustrate how ChatGPT can rephrase radiology reports into patient-friendly language, and Kung et al. ( kung2023 ) report its near-passing performance on the USMLE, suggesting a role in medical education. Johnson et al. ( johnson2023 ) confirm LLMs’ general clinical knowledge but emphasize the need for domain-specific fine-tuning. Building on these advances, Strong et al. ( strong2024deferral ) introduce guided deferral systems that prompt AI to flag uncertain cases for human review, striking a balance between efficiency and safety. Meanwhile, Biswas and Talukdar ( biswas2024clinicaldoc ) demonstrate how generative AI can draft SOAP notes from clinician-patient dialogs, freeing practitioners to focus on care without sacrificing documentation quality.

 

 
 
 The healthcare field showcases both the promise and limitations of LFMs. They can speed up the reporting and suggest diagnoses, but clinical judgment and empathy cannot be replaced. This creates a high-stakes partnership where the real challenge is calibrating trust: doctors need systems that explain their reasoning, show confidence levels, and make limitations clear. Only then can AI act as a reliable partner rather than an opaque black box.

 
 
 Figure 9 . An example of how multimodal foundation models and large language models can be used for enhancing generalization and robustness in autonomous driving, as studied by Wang et al. ( wang2023drive ) . They develop pixel/patch-aligned feature descriptors and latent space simulation, enriched with language modality, suggesting potential for optimizing the training and debugging processes for end-to-end learning-based control. The RGB images in this figure are taken from the KITTI dataset ( Geiger2012CVPR ) for representation purposes. An autonomous driving pipeline combines multimodal visual features with language-based latent-space simulation to produce vehicle control. Three road-scene RGB images are processed by a multimodal foundation model. Its representation is used for patch-wise feature extraction and for language-augmented latent-space simulation. In the simulation example, an LLM connects concepts from an initial rural scene, including car, road, and tree, to concepts in a target urban scene, including building and supermarket. The resulting visual and language-augmented representations are passed to a policy enforcer network, which produces the vehicle control output represented by a steering wheel. 
 
 
 

### 5.2. Autonomous Vehicles

 
 Autonomous vehicles depend on fluent human-AI collaboration, especially when shifting control or interpreting complex scenarios. Atakishiyev et al. ( Atakishiyev2021 ) showcase how advanced sensor fusion and decision pipelines reduce collision risk, while Lv et al. ( Lv2021 ) introduce hybrid control schemes that smoothly transfer authority back to drivers during edge cases. Zhou and Chen’s uncertainty-aware framework ( Jianlong2019 ) further builds driver confidence by quantifying AI hesitation, and Fuchs et al. ( Fuchs2023 ) employ reinforcement learning to optimize how tasks are split between human supervisors and automated modules.

 
 
 LLMs are now illuminating new collaboration pathways. Yang et al. ( yang2024 ) demonstrate conversational diagnostics that translate raw system logs into clear guidance, improving driver situational awareness on the fly. Park et al. ( park2024 ) present VLAAD, a multimodal LLM that fuses sensor feeds with natural-language queries, closing the gap between machine perception and human intent. Cui et al. ( cui2024drive ) push this further: drivers can adjust planned trajectories via plain-English voice commands, blending AI planning with human oversight. Tian et al. ( tian2024critical ) introduce CRITICAL, an LLM-guided scenario generator that helps human testers uncover rare failure modes before real-world deployment. Complementing these advances, Wang et al. ( wang2023empowering ) explore LLMs as high-level behavior planners, Wen et al. ( wen2023 ) leverage vision-language models like GPT-4V for richer scene interpretation, and Wang et al. ( wang2023drive ) demonstrate end-to-end driving pipelines built on foundation models that use language-driven simulation for policy debugging (Fig. 9 ).

 

 
 
 While promising, autonomous driving research reveals the gap between the dream of full autonomy and the reality of unpredictable edge cases. LFMs help by translating complex logs into natural language, boosting situational awareness, but they do not completely solve the hardest part: safe, seamless control handoffs. The open challenge is designing intuitive mechanisms that manage driver workload and ensure readiness, bridging the space between machine perception and human action.

 
 
 

### 5.3. Surveillance and Security

 
 Modern surveillance and security operations take advantage of effective human-AI collaboration. Vision-based AI and networked cameras have become vital tools for Cyber Security Incident Response Teams (CSIRTs) ( killcrece_2003 ; Pazho2023 ) , but striking the right balance of AI autonomy is key. Hauptman et al. ( Hauptman2023 ) surveyed 103 practitioners and interviewed 22 more, finding that higher autonomy is more acceptable during routine, predictable phases while greater human oversight is preferred during higher-stakes phases such as containment and recovery. Similarly, Guo et al. ( Guo2018 ) demonstrate how crowd-powered camera networks, enhanced by human verification, deliver rapid situational awareness without sacrificing accuracy.

 
 
 Recent work transfers LFMs into the surveillance domain. Chen et al.’s VideoLLM framework ( chen2023 ) shows how LLMs can interpret video streams, tagging suspicious behaviors in plain language for human analysts. Jain ( jain2023 ) further integrates ChatGPT into security workflows, automating report drafts while retaining human review to catch context-specific nuances.

 
 
 Looking ahead, adaptive human-AI teamwork strategies promise to transform security operations. Chhetri et al. ( baruwalchhetri2024alertfatigue ) present a human-in-the-loop deferral framework that routes uncertain alerts to analysts and continuously refines its deferral policy based on their feedback, cutting triage workload by over 30 percent. Oliver et al.’s Carbon Filter ( oliver2024carbonfilter ) applies large-scale clustering and fast search to group similar alerts, then leverages human corrections to recalibrate clusters in real time, achieving a six-fold boost in signal-to-noise ratio. These interactive dashboards and feedback loops illustrate how ongoing human-AI collaboration can streamline alert triage, uphold detection accuracy, and sustain analyst trust under high-volume conditions.

 

 
 
 In security, speed and scale cut both ways. LFMs can identify alerts and draft reports faster than ever, but the flood of information risks overwhelming human analysts. The challenge is moving past simple filtering to real sensemaking; systems that group related events, infer intent, and present a clear story. When done right, AI shifts from being just a filter to a true partner in strategic defense.

 
 
 

### 5.4. Games

 
 Games offer dynamic testbeds for human-AI collaboration, where mixed-initiative workflows and natural language play crucial roles. We highlight two key trends: LLMs as active game partners and LLMs as content creators.

 
 

#### 5.4.1. LLM-Based AI as Players 

 
 Early game AIs, like the scripted bots in Mario Kart 8 Deluxe ( MarioKart8Deluxe2017 ) , excel at rule-based play but lack flexibility and language skills. LLMs bridge this gap: Frans ( frans2021 ) used GPT-2 to play ‘AI Charades,’ parsing and generating expressive clues, and Xu et al. ( xu2023werewolf ) demonstrated strategic AI agents in the social deduction game Werewolf. Yet reliability remains a concern—Sobieszek and Price ( sobieszek2022 ) document occasional hallucinations, and Akata et al. ( akata2023 ) show mixed results when LLMs navigate moral dilemmas. More recently, Sidji et al. ( sidji2024codenames ) find that LLMs partnered with humans in Codenames improve clue precision and team performance, underscoring the promise of human–AI synergy in cooperative play.

 
 
 

#### 5.4.2. LLMs for Content Generation in Games 

 
 LLMs also enhance game design by automating narrative and level creation. Todd et al. ( todd2023 ) generate coherent game levels via prompt-driven architectures, while Vartinen et al. ( vartinen2022 ) craft dynamic quest stories that adapt to player choices. InstructRL ( hu2023 ) integrates human feedback loops to fine-tune tutorial dialogues, ensuring clarity and engagement. However, ethical concerns around bias and coherence persist ( sobieszek2022 ) . White et al. ( white2024communicate ) address cultural variation in narrative generation with RSA+C3, improving cross-cultural teamwork in Codenames through pragmatically tailored prompts.

 

 
 
 Hence, in gaming, LFMs can generate endless content and act as dynamic partners, but their outputs often lack coherence and can carry bias. The challenge is building workflows where humans stay in the loop; guiding narratives, enforcing fairness, and adding cultural sensitivity. That way, AI expands creative potential without losing authorial control or ethical oversight.

 
 
 
 

### 5.5. Education

 
 Teaching and learning are being reshaped by human-AI collaborative teams that adapt to each student’s needs in real time. Early Intelligent Tutoring Systems (ITSs) like Carnegie Learning’s Mika platform ( nwana1990 ; vanlehn2011 ) have raised exam scores and reduced dropouts, yet the opacity of their models has sparked equity concerns ( whitaker2013 ) . Platforms such as Century Tech and Fishtree embed adaptive dashboards driven by student data to guide teacher interventions ( kasinathan2017 ) . Virtual learning agents, most notably Georgia Tech’s Jill Watson ( eicher2018 ; hartle2019 ) , field administrative questions and peer-collaboration tasks, freeing instructors for deeper engagement but risking parasocial ties. Automated essay scoring systems ( dikli2006 ) speed grading and align closely with human raters, though they still fall short on subtle expression.

 
 
 Large language models extend this collaboration by engaging learners in natural dialogue and generating tailored content. Extance ( extance2023 ) shows AI chatbots boosting engagement through instant, contextual feedback, while Milano et al. ( milano2023 ) warn that LLM deployment must account for environmental and ethical impacts. Recent studies underscore the need for transparency and robust data governance in classroom AI applications ( rose2023a ; hellas2023 ; chang2023 ) . Building on these insights, Kong et al. ( kong2025synergy ) introduce a Synergy Degree Model to quantify the quality of human-AI interaction in hybrid learning environments, guiding iterative refinement of teaching practices. Schotter et al. ( schotter2025spiral ) demonstrate that integrating generative AI into creative media courses significantly enhances student self-efficacy and career expectations, highlighting the lasting value of continuous human–AI collaboration in curriculum design.

 

 
 
 AI in education promises personalized, scalable tutoring, but also risks inequity and para-social attachments. LFM tutors can adapt quickly, yet their opacity makes it hard for teachers to see why a recommendation was made. The challenge is to build teacher-centric tools, such as transparent dashboards and override controls, that let educators stay in charge. AI should support classroom practice, not dictate it.

 
 
 

### 5.6. Accessibility

 
 Assistive AI thrives on tight human-AI feedback loops that personalize support for diverse needs. Kumar et al. ( kumar2022 ) introduce a vision-based navigation assistant for the visually impaired that refines its obstacle detection models through user corrections in real time. Ozarkar et al. ( Ozarkar2020 ) combine audio-visual recognition with interactive prompts to help deaf users verify and improve lip-reading accuracy. Khan et al. ( Khan2020 ) demonstrate a compact visual aid that lets blind users flag misdetections, enabling the system to learn new object categories on the fly. Wen et al. ( Wen2021 ) embed a human-in-the-loop sign language recognizer that adapts to individual signing styles, boosting sentence-level accuracy over static models. Ghazal et al. ( Ghazal2021 ) further show how eldercare robots can request for verbal feedback during shopping tasks, tailoring their assistance to each user’s pace and preferences.

 
 
 Large language models are opening fresh avenues for accessibility, but they also bring new challenges. Taheri et al. ( taheri2023 ) use text-to-image LLMs to let motor-impaired artists sketch via simple prompts, iterating with user feedback for stylistic control. Gadiraju et al. ( gadiraju2023 ) warn that without inclusive training data, LLMs may perpetuate stereotypes against disabled communities, highlighting the need for continual human audits. Recent prototypes push these ideas further: Brilli et al. ( brilli2024airis ) present AIris, an AI-powered wearable that narrates scenes and ask for corrective cues, enabling blind users to teach the system new objects. Tokmurziyev et al. ( tokmurziyev2025llmglasses ) develop LLM-Glasses, which fuse GPT-driven reasoning with haptic feedback and on-device corrections, achieving over 90 percent navigation accuracy in dynamic environments.

 

 
 
 While super-helpful, accessibility with AI highlights the gap between universal tools and deeply personal needs. LFMs can narrate scenes or power communication aids, but one-size-fits-all rarely works. The challenge is creating co-adaptive, customizable systems that learn from user feedback. With humans in the loop to guide and correct, assistive AI can move beyond functional to truly empowering .

 
 
 
 

## 6. Open Challenges and Future Research Directions

 
 This survey has traced the rapid evolution of Human-AI Collaboration in the era of Large Foundation Models. Progress has been remarkable, but there are still deep challenges to solve if these partnerships are to be effective, fair, and trustworthy. We group these open problems into four themes.

 
 

### 6.1. Scalable and Diverse Human Guidance

 
 Foundation models depend heavily on human feedback, yet collecting it at scale while preserving diversity is difficult. Techniques like RLHF demonstrate the value of preference-based training, but they risk aligning models to the views of narrow groups, amplifying societal bias. The path forward lies in finding ways to make feedback both scalable and representative, seeking out underrepresented voices, developing mechanisms to reconcile conflicting preferences, and training human evaluators to act as skilled, adversarial testers rather than passive labelers.

 
 
 

### 6.2. Fluid but Controllable Interaction

 
 Conversational interfaces have made collaboration with AI feel more natural, but they come at the cost of precision and control. In high-stakes contexts, natural language alone can leave users uncertain about what the system is doing and when they should intervene. This issue becomes sharper in agentic systems, where the AI may plan, call tools, and execute several steps before returning a result. The main risk is that errors can accumulate silently across a long action chain: a wrong assumption early in the plan can lead to incorrect tool calls, file edits, messages, or decisions later. This makes oversight harder than in ordinary AI augmentation, where the user usually checks a single output. Future systems, therefore, need interfaces that expose the agent’s plan, show intermediate progress, allow users to pause or redirect execution, and place approval gates before irreversible or high-risk actions. Evaluation methods must also evolve to measure not just task success, but the quality of collaboration, capturing trust, workload, and resilience.

 
 
 

### 6.3. Accountability and Governance

 
 The speed of deployment of large models has far overtaken the pace of governance, leaving a troubling gap in accountability. When human-AI teams cause harm, the blurred lines of responsibility make it difficult to determine what went wrong.
Agentic workflows also make accountability harder. When an AI system decomposes a task, calls tools, edits files, or communicates with other systems, harm may result from a chain of small intermediate choices rather than a single final output. Effective governance, thus, requires interfaces and protocols that make the agent’s process inspectable and interruptible. Moving forward requires embedding fairness and privacy directly into collaborative systems, while also creating auditable records of decisions so that responsibility can be clearly assigned. Governance must move beyond abstract principles to concrete, enforceable standards that balance innovation with accountability.

 
 
 

### 6.4. Contextual Grounding in High-Stakes Domains

 
 Finally, this survey underlines that foundation models are not one-size-fits-all solutions. In domains like healthcare, security, or education, simply deploying a general-purpose model is not only ineffective but potentially dangerous. The last mile of research must focus on contextual grounding: calibrating professional trust, ensuring that experts understand model reasoning and limits, scaffolding human skill rather than replacing it, and allowing systems to adapt to the unique needs of individuals, particularly in accessibility settings.

 
 
 
 

## 7. Conclusion

 
 Large Foundation Models have moved Human-AI Collaboration from the margins to the center of AI research. Building effective and responsible partnerships is not about raw model power but about design choices across the whole lifecycle: how data is curated, how objectives are shaped, how interfaces are built, and how governance is enforced. Across domains such as healthcare, education, security, and accessibility, these collaboration patterns only succeed when carefully adapted to context. Simply deploying a general model is not enough, and in high-stakes settings, it can be risky. Looking forward, the central challenges look clear: scaling human guidance without losing diversity, creating interfaces that are both natural and controllable, embedding accountability into fast-moving systems, and grounding models in the realities of each domain. The future of AI will not be defined by autonomy alone, but by the quality of the partnerships we build with it.

 
 
 

## References