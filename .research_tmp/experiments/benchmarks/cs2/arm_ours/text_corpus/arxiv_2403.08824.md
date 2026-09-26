Computational Analysis of Stress, Depression and Engagement in Mental Health: A Survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2403.08824v2 [cs.HC] 25 Mar 2025 
 
 

# Computational Analysis of Stress, Depression and Engagement in Mental Health: A Survey

 
 
 Puneet Kumar
 
 
    
 Alexander Vedernikov
 
 
    
 Yuwei Chen
 
 
    
 Wenming Zheng
 
 
    
 Xiaobai Li*
 
 † † thanks: *Corresponding Author. † † thanks: P. Kumar and A. Vedernikov are with the Center for Machine Vision and Signal Analysis, University of Oulu, Finland. Email: puneet.kumar@oulu.fi, aleksandr.vedernikov@oulu.fi. † † thanks: Y. Chen is with Hangzhou Institute for Advanced Study, University of Chinese Academy of Sciences, Hangzhou, China. Email: yuwei.chen@ucas.ac.cn. † † thanks: W. Zheng is with the Key Laboratory of Child Development and Learning Science (Southeast University), Ministry of Education and also with the School of Biological Science and Medical Engineering, Southeast University, Nanjing 210096, China. Email: wenming_zheng@seu.edu.cn † † thanks: X. Li is with the State Key Laboratory of Blockchain and Data Security, Zhejiang University, Hangzhou, China and the Center for Machine Vision and Signal Analysis, University of Oulu, Finland. E-mail: xiaobai.li@zju.edu.cn † † thanks: Manuscript Received: Mar 2025. 

 Abstract 
 
 Analysis of stress, depression and engagement is less common and more complex than that of frequently discussed emotions such as happiness, sadness, fear and anger. The importance of these psychological states has been increasingly recognized due to their implications for mental health and well-being. Stress and depression are interrelated and together they impact engagement in daily tasks, highlighting the need to explore their interplay. This survey is the first to simultaneously explore computational methods for analyzing stress, depression and engagement. We present a taxonomy and timeline of the computational approaches used to analyze them and we discuss the most commonly used datasets and input modalities, along with the categories and generic pipeline of these approaches. Subsequently, we describe state-of-the-art computational approaches, including a performance summary on the most commonly used datasets. Following this, we explore the applications of stress, depression and engagement analysis, along with the associated challenges, limitations and future research directions.

 
 
 
 Index Terms:  Affective Computing, Health Informatics, Mental Health Applications, Machine Learning, Psychological State Analysis.

 
 

## I Introduction 

 
 Affective Computing involves the development of computational approaches to analyze a broad spectrum of psychological states [ 1 ] . Psychological State is a broad term encompassing various mental conditions related to affect and cognition [ 2 ] . Affect refers to the experience of feeling or emotion, including broader concepts such as Emotion , Mood , Sentiment and Opinion . These aspects collectively characterize how individuals experience and express their emotional states. In contrast, Cognition involves the mental processes of acquiring knowledge and understanding through thought, experience and the senses [ 3 ] . This includes functions like perception, memory and judgment, crucial for processing and interpreting information. Emotion is a manifestation of affect marked by complex mental states and physiological responses, with theories that categorize emotions through dimensions like valence and arousal or into discrete classes such as happiness, sadness, fear and anger [ 4 , 5 ] . Mood is a more lasting but less intense expression of affect, whereas emotions give rise to Sentiments over time, which are basic mental attitudes [ 6 ] . The sentiments subsequently lead to Opinions , which are personal interpretations shaped by one’s experience [ 7 ] .

 
 
 Fig. 1 : Illustration showing the interaction between basic emotions and psychological states: the left image shows a boy happy and engaged with his reading, while the right image shows him sad but still engaged, demonstrating that psychological states can coexist with different emotions. This image was created with DALL·E 2 . 
 
 
 There are six basic emotions: anger, surprise, disgust, happiness, fear and sadness, as proposed by Paul Ekman [ 4 ] . These emotions are extensively studied and universally recognized in different cultures and ethnicities. In contrast, there are complex psychological states such as stress, guilt, depression, shame, pride, curiosity, empathy, envy, engagement, etc., that are not as frequently explored in the literature [ 8 ] . Fig. 1 demonstrates how basic emotions interact with complex psychological states, illustrating that various psychological states can coexist with diverse emotional contexts. This survey explores a broad range of psychological states that extend beyond basic emotions, with a focus on those relevant to mental health analysis. Appraisal theories, as proposed by Scherer [ 9 ] and Roseman [ 10 ] , offer a richer framework for understanding the complexities of emotional states beyond simple discrete categories. According to these models, emotions are appraised through multiple dimensions, which influence how they are perceived and experienced. Furthermore, embodying emotion theory [ 11 ] suggests that emotions involve bodily responses, while constructivist theory [ 12 ] argues that emotions are constructed from core psychological systems, rather than triggered by external events. This perspective advocates a deeper exploration of less explored psychological states.

 
 
 This survey reviews computational approaches for analyzing stress, depression and engagement, which are interrelated and play a significant role in mental health analysis [ 13 ] . Stress disrupts attentional networks and motivation [ 3 ] , while the lack of interest associated with depression further compromises an individual’s ability to remain engaged [ 14 ] . Stress and depression reduce daily efficiency and task performance [ 15 , 16 ] . Several computational studies have highlighted the correlation among stress, depression and engagement. For instance, Pizzagalli et al. [ 14 ] have presented an integrated model in which stress disrupts reward processing and leads to anhedonia which is a core mechanism linking stress to depression. In another work, Slavich and Irwin [ 17 ] have proposed a social signal transduction theory that explains how stress-induced inflammation contributes to the development of major depressive disorder, thereby impairing cognitive functions essential for maintaining engagement. Moreover, a longitudinal study by Innstrand et al. [ 18 ] demonstrated that reduced work engagement is significantly associated with increased symptoms of depression and anxiety, underscoring the dynamic interplay among these psychological states.

 
 
 Neurophysiological studies have also indicated the interconnected nature of stress, depression and engagement in cognitive and emotional processes, underscoring their importance for mental health and well-being [ 19 ] . Chronic stress has been shown to induce key neurobiological changes that affect synaptic integrity and neurotransmission in crucial regions such as the limbic system and prefrontal networks, ultimately leading to neuroendocrine dysfunction and an increased vulnerability to depression, as detailed by Krishnan et al. [ 20 ] and expanded by Lupien et al. [ 19 ] . Similarly, Arnsten [ 3 ] and Liston et al. [ 21 ] have highlighted that stress negatively impacts the prefrontal cortex, impairing cognitive functions such as executive control and attention which are essential for sustaining engagement in demanding tasks. In addition, Pizzagalli et al. [ 14 ] have shown that stress-related changes in the brain’s reward circuits, particularly in the ventral striatum and medial prefrontal cortex, result in anhedonia and motivational deficits typical of depression, further reducing engagement by diminishing responsiveness to rewarding stimuli. Heller [ 22 ] has noted that depression is correlated with a decreased ability to maintain activation in frontostriatal networks, which are crucial for generating and sustaining positive emotions and engagement. Lastly, Slavich and Irwin [ 17 ] have proposed a model linking psychosocial stress to neuroimmune alterations affecting mood- and cognition-related neural circuits, thereby further bridging stress and depression on a neurophysiological level.

 
 
 While previous surveys have individually addressed stress ( [ 23 , 24 , 25 ] ), depression ( [ 26 , 27 , 28 ] ) and engagement ( [ 29 , 30 , 31 , 32 ] ), there is still a notable gap in understanding their interdependencies. None have simultaneously explored the methodologies, applications and challenges encompassing computational models for all three of these psychological states, leaving an opportunity to investigate how they intersect and influence one another. Stress has been examined from multiple perspectives, including its implications in workplace environments [ 25 ] , its effects on drivers [ 24 ] and its broader psychological impacts [ 23 ] . When examining depression, past surveys have delved into approaches using Electroencephalogram signals for deeper insights [ 26 ] , methods based on audio-visual information [ 27 ] and the crucial connections between depression and patient engagement in healthcare settings [ 28 ] . Similarly, discussions on engagement have spanned diverse scenarios such as academic engagement [ 29 ] , engagement in workplace contexts [ 30 ] , inpatient care [ 32 ] and human-machine interactions [ 31 ] . Despite the breadth of existing surveys on stress, depression and engagement, no unified work has yet examined all three together. This paper addresses that gap by presenting a comprehensive survey of computational approaches for analyzing these three psychological states in tandem, thereby providing a broader perspective on their collective significance.

 
 
 Fig. 2 : Publication trends in stress, depression and engagement from 1996 to 2024 indicate a growing interest in these areas. This underscores the increasing importance of computational methods in addressing mental health challenges and enhancing well-being. 
 
 
 The computational approaches for analyzing these psychological states have seen substantial growth, as depicted in Fig. 2 , with detailed trends and methodological evolutions discussed in Section IV . This paper surveys these approaches in a comprehensive manner, emphasizing their interconnectedness and the way advances in one area have influenced the others. For our literature review, we employed advanced queries in the Scopus database, targeting publications from 1994 to 2024 in top-tier journals and highly ranked conferences as per SCI and Qualis rankings. This search extended to include computational, psychological and neuroscience studies relevant to stress, depression and engagement, ensuring a broad and multi-disciplinary perspective. We manually reviewed the final list of papers to remove any irrelevant or redundant publications. Additionally, we made detailed notes on modalities, datasets used, contributions, methods, code availability, evaluation metrics, results and applications mentioned in these papers and used them during manuscript writing.

 
 
 To aid the organization and flow of this survey paper, a detailed taxonomy outlining the computational analysis of stress, depression and engagement is presented in Table I . It summarizes various emotion categories, input modalities, computational approaches and applications of stress, depression and engagement analysis. The categorical and dimensional emotion classes are introduced in Section I . Various datasets and input modalities for stress, depression and engagement analysis have been described in detail in Sections II and Section  III respectively. Section  IV describes various computational approaches for stress, depression and engagement analysis along with their generic framework and state-of-the-art. The applications of these approaches have been discussed in Section V . Section VI highlights challenges and future directions and Section VII concludes the paper.

 
 

 TABLE I : A taxonomy for computational analysis of stress, depression and engagement. The relevant research papers are presented based on emotion categories, input modalities (detailed in Section III ), computational approaches (discussed in Section IV.2 ) and applications (mentioned in Section V ). Acronyms used include Machine Learning (ML), Deep Learning (DL), Convolutional Neural Network (CNN), Long Short Term Memory Network (LSTM), Gated Recurrent Unit (GRU), Support Vector Machine (SVM), K-Nearest Neighbour (KNN), Multimodal Learning (MM) and Bidirectional Encoder Representations from Transformers (BERT). 
 
 
 
 Taxonomy | 
 | 
 | 
 | 

 
 Emotions | 
 | 
 | 

 
 Categorical 
 
 [ 10 , 33 , 34 , 35 , 36 , 37 , 38 , 39 , 40 , 41 , 42 , 43 , 31 , 44 , 45 , 46 , 47 , 48 , 49 ] 
 | 
 | 

 
 Dimensional 
 
 [ 12 , 50 , 51 , 52 , 53 , 54 , 23 , 55 , 56 , 57 , 58 , 24 , 59 , 60 , 61 , 62 ] 
 | 
 | 

 
 
 
 Inputs 
 (Sec. III ) 
 | 
 | 
 | 

 
 Visual 
    Facial Features : [ 63 , 64 , 40 , 35 , 46 , 65 , 66 ] | 
 | 

 
         Action Units : [ 67 , 34 , 68 , 69 ] | 
 | 

 
         Eye Tracking Metrics : [ 70 , 71 , 44 , 72 ] | 
 | 

 
         Body Dynamics : [ 73 , 74 , 75 , 76 ] | 
 | 

 
         Micro-Gestures : [ 50 , 48 , 77 , 64 ] | 
 | 

 
 Physiological 

 
 [ 51 , 52 , 53 , 23 , 55 , 56 , 57 , 58 , 60 , 61 , 78 , 79 , 80 , 81 , 82 , 83 , 26 , 84 , 85 , 86 , 87 ] 
 | 
 | 

 
 Audio [ 88 , 89 , 90 , 39 , 91 , 33 , 92 , 93 ] | 
 | 

 
 Text 
 
 [ 59 , 94 , 95 , 96 , 97 , 42 , 43 , 98 , 99 , 36 , 37 , 87 , 100 ] 
 | 
 | 

 
 Motion 
 
 [ 101 , 102 , 60 ] 
 | 
 | 

 
 Multimodal 
 
 [ 65 , 103 , 104 , 105 , 106 , 62 , 107 , 108 , 109 , 110 , 74 , 111 , 112 , 113 , 114 , 38 , 115 , 116 , 117 , 118 , 93 , 119 , 120 , 121 , 122 , 123 , 124 , 125 , 93 ] 
 | 
 | 

 
 
 
 Approaches 
 (Sec. IV ) 
 | 
 | 
 | 

 
     ML 
    Logistic Regression : [ 95 , 85 , 126 ] | 
 | 

 
          Decision Trees : [ 51 , 56 , 57 , 93 , 127 ] | 
 | 

 
          Ensemble Methods : [ 51 , 57 , 113 , 128 , 129 , 93 , 130 , 100 ] | 
 | 

 
          Naive Bayes : [ 57 , 95 , 126 , 93 , 86 ] | 
 | 

 
          SVM and KNN : [ 57 , 131 , 85 , 86 ] | 
 | 

 
          Dimensionality Reduction : [ 128 , 114 , 34 , 127 , 125 , 132 ] | 
 | 

 
          Clustering : [ 42 , 117 , 116 ] | 
 | 

 
     DL 
    CNNs : [ 128 , 133 , 134 , 135 , 136 , 137 , 108 , 107 , 109 , 111 , 90 , 88 , 93 , 130 ] | 
 | 

 
          RNN / LSTM / GRU : [ 135 , 109 , 90 , 88 , 92 ] | 
 | 

 
          ResNet : [ 106 , 50 , 40 , 41 , 86 ] | 
 | 

 
          Graph Neural Networks : [ 138 , 139 , 38 , 140 , 141 ] | 
 | 

 
          Attention Mechanism : [ 34 , 142 , 108 , 112 , 114 , 143 , 35 , 122 ] | 
 | 

 
          Transformer / BERT : [ 42 , 43 , 87 , 125 , 122 ] | 
 | 

 
          Autoencoder : [ 89 , 38 , 37 , 100 , 132 ] | 
 | 

 
          Interpretable Deep Networks : [ 128 , 35 , 43 , 92 ] | 
 | 

 
          Federated Learning : [ 36 , 37 , 144 ] | 
 | 

 
 | 
     MM 
   Multimodal Fusion : [ 112 , 109 , 74 , 113 , 41 , 108 , 107 , 117 , 112 , 93 , 125 ] | 
 | 

 
 | 
          Hybrid Models : [ 134 , 135 , 137 , 108 , 107 , 109 , 111 , 90 , 88 , 130 ] | 
 | 

 
 | 
       Transfer Learning : [ 58 , 48 , 103 , 88 , 39 , 45 , 88 , 145 ] | 
 | 

 
 | 
       Self-Supervised Learning : [ 73 , 108 , 63 , 84 , 112 , 125 ] | 
 | 

 
 | 
      Human-Centered Computing : [ 146 , 134 , 96 , 24 , 147 ] | 
 | 

 
 | 
 Wearable Technologies : [ 114 , 53 , 53 , 148 , 56 , 60 , 79 , 130 ] | 
 | 

 
 | 
      Personalized Learning : [ 149 , 56 , 146 , 134 ] | 
 | 

 
 | 
     Blended Learning : [ 31 , 124 , 150 , 66 ] | 
 | 

 
 | 
 Cloud-Edge Computing : [ 84 , 151 , 127 , 144 ] | 
 | 

 
 
 
 Applications 
 (Sec. V ) 
 | 
 | 
 | 

 
 Technological Solutions for Mental Health :
 [ 56 , 131 , 152 , 60 , 148 , 79 , 53 , 37 , 85 , 86 , 87 , 125 ] | 
 | 

 
 Workplace and Occupational Well-being :

 
 [ 153 , 55 , 24 , 154 , 82 , 155 , 156 , 157 , 85 , 126 ] 
 | 
 | 

 
 Detecting Mental Health Disorders :
 [ 113 , 97 , 158 , 15 , 87 , 85 , 86 , 125 ] | 
 | 

 
 Health and Behaviour Monitoring :
 [ 106 , 24 , 159 , 36 , 38 , 40 , 109 , 93 ] | 
 | 

 
 Treatment Planning for Mental Health Disorders :

 
 [ 41 , 143 , 61 , 154 , 79 , 111 , 53 , 51 ] 
 | 
 | 

 
 Education and Learning Analytics :
 [ 46 , 70 , 124 , 41 , 49 , 146 , 75 , 134 , 160 , 161 , 162 , 66 , 150 ] | 
 | 

 
 Gaming and Entertainment :
 [ 108 , 107 , 163 , 83 , 137 , 153 ] | 
 | 

 
 Human-Computer Interaction :

 
 [ 40 , 41 , 44 , 70 , 33 , 75 , 134 , 98 , 114 , 117 , 152 , 42 , 147 ] 
 | 
 | 

 
 Ethics and Privacy Preservation :
 [ 95 , 60 , 74 , 162 , 132 , 144 ] | 
 | 

 
 Policy Making and Social Support :
 [ 75 , 142 , 87 , 100 ] | 
 | 

 
 
 
 

## II Datasets 

 
 Table II summarizes stress, depression and engagement datasets, with sizes given in hours except for unimodal text-only datasets where duration is not applicable. Their details are discussed in the following sections and sample inputs are shown in Fig. 3 .

 
 
 TABLE II : Summary of datasets for Stress, Depression and Engagement analysis. Here ‘A’, ‘M’, ‘P’, ‘T’ and ‘V’ represent ‘Audio’, ‘Motion’, ‘Physiological’, ‘Textual’ and ‘Visual’ modalities, respectively and
‘ GT ’ specifies the type of Ground Truth label used, with ‘HA’ for Humanly Annotated, ‘TD’ for Task Determined, ‘CA’ for Clinically Assessed and ‘SR’ for Self Report. 
 
 
 
 Name | 
 Year | 
 Focus Area | 
 Size (hours) | 
 Subjects | 
 Modalities | 
 GT | 

 
 Stress | 

 
 StressID [ 164 ] | 
 2023 | 
 Multiple Stimuli Stress | 
 39 | 
 65 Adults | 
 APV | 
 SR | 

 
 SMG [ 50 ] | 
 2023 | 
 Micro-Gestures | 
 8.14 | 
 40 Adults | 
 V | 
 HA | 

 
 MAUS [ 165 ] | 
 2021 | 
 Workload Stress | 
 12.83 | 
 22 Adults | 
 P | 
 TD+SR | 

 
 ULM-TSST [ 105 ] | 
 2021 | 
 Public Stress | 
 5.78 | 
 105 Participants | 
 P | 
 SR | 

 
 MuSe-CaR [ 166 ] | 
 2021 | 
 Driver Stress | 
 40.2 | 
 38 Drivers | 
 APTV | 
 HA | 

 
 VerBIO [ 167 ] | 
 2021 | 
 Bio-behaviour | 
 180 | 
 55 Students | 
 APT | 
 SR | 

 
 CLAS [ 168 ] | 
 2019 | 
 Workplace Stress | 
 31 | 
 62 Adults | 
 APT | 
 TD+SR | 

 
 DASPS [ 169 ] | 
 2019 | 
 Anxiety Analysis | 
 1.15 | 
 23 Participants | 
 P | 
 SR | 

 
 WESAD [ 60 ] | 
 2018 | 
 Wearable Stress | 
 13.38 | 
 15 Adults | 
 PM | 
 SR | 

 
 Passau-SFCH [ 170 ] | 
 2017 | 
 Stress in Sports | 
 11 | 
 10 Footballers | 
 ATV | 
 HA | 

 
 DRIVEDB [ 171 ] | 
 2000 | 
 Driver Stress | 
 21.97 | 
 17 Drivers | 
 P | 
 ED | 

 
 Depression | 

 
 MPDD [ 172 ] | 
 2025 | 
 Depression Personality | 
 9.68 | 
 228 Participants | 
 ATV | 
 CA | 

 
 CMDC [ 173 ] | 
 2023 | 
 Semi-structured Interviews | 
 1.41 | 
 167 Adults | 
 ATV | 
 CA | 

 
 MMDA [ 174 ] | 
 2022 | 
 Clinical Interviews | 
 48.05 | 
 1025 Participants | 
 ATV | 
 CA | 

 
 EATD [ 175 ] | 
 2022 | 
 Short Q A Interviews | 
 2.26 | 
 162 Participants | 
 AT | 
 SR | 

 
 D-Vlog [ 176 ] | 
 2022 | 
 Vlog Recordings | 
 160 | 
 816 Participants | 
 AV | 
 HA | 

 
 MODMA [ 177 ] | 
 2020 | 
 Mental Disorder Analysis | 
 49.5 | 
 55 Participants | 
 AP | 
 CA | 

 
 Chi-Mei [ 178 ] | 
 2020 | 
 Mood Database | 
 6.8 | 
 11 Participants | 
 A | 
 CA | 

 
 Extended DAIC [ 179 ] | 
 2019 | 
 Clinical Interviews | 
 71.38 | 
 185 Adults | 
 ATV | 
 SR | 

 
 SH2 [ 91 ] | 
 2018 | 
 Phone Ctterances | 
 16 | 
 887 Participants | 
 A | 
 TD+CA | 

 
 UM Suicidality [ 180 ] | 
 2018 | 
 Suicidality Reports | 
 – | 
 934 Subjects | 
 T | 
 HA+CA | 

 
 eRisk [ 181 ] | 
 2017 | 
 Health Risk Detection | 
 – | 
 4427 Subjects | 
 T | 
 HA+SR | 

 
 DAIC-WOZ [ 182 ] | 
 2016 | 
 Clinical Interviews | 
 50.21 | 
 189 Adults | 
 ATV | 
 SR | 

 
 BlackDog [ 183 ] | 
 2016 | 
 Open Ended Questions | 
 8.55 | 
 130 Participants | 
 A | 
 CA | 

 
 CLPsych [ 184 ] | 
 2015 | 
 Mental Health Posts | 
 – | 
 1989 Subjects | 
 T | 
 HA+SR | 

 
 AVEC 2014 [ 185 ] | 
 2014 | 
 Emotion Challenge | 
 4.52 | 
 58 Adults | 
 AV | 
 SR | 

 
 AVEC 2013 [ 186 ] | 
 2013 | 
 Emotion Challenge | 
 39.5 | 
 58 Adults | 
 AV | 
 SR | 

 
 RECOLA [ 187 ] | 
 2013 | 
 Remote Collaboration | 
 9.5 | 
 46 Participants | 
 APV | 
 HA+SR | 

 
 Pittsburgh [ 188 ] | 
 2012 | 
 Clinical Interviews | 
 3.61 | 
 49 Participants | 
 A | 
 CA | 

 
 Engagement | 

 
 DREAMS [ 189 ] | 
 2024 | 
 Engagement Attention | 
 8.68 | 
 32 Adults | 
 V | 
 SR | 

 
 EngageNet [ 190 ] | 
 2023 | 
 Student Engagement | 
 31 | 
 127 Adults | 
 V | 
 HA+SR | 

 
 PAFE [ 49 ] | 
 2022 | 
 Student Engagement | 
 15 | 
 15 Adults | 
 V | 
 SR | 

 
 VRESEE [ 46 ] | 
 2022 | 
 Student Engagement | 
 9.79 | 
 88 Adults | 
 V | 
 HA+SR | 

 
 FaceEngage [ 163 ] | 
 2019 | 
 Gameplay Engagement | 
 2.18 | 
 25 Adults | 
 V | 
 HA | 

 
 EngageWild [ 160 ] | 
 2018 | 
 Student Engagement | 
 16.5 | 
 91 Adults | 
 V | 
 HA | 

 
 UE-HRI [ 118 ] | 
 2017 | 
 Human-Robot Interaction | 
 24.98 | 
 54 Adults | 
 AV | 
 HA | 

 
 MHHRI [ 115 ] | 
 2017 | 
 Human-Robot Interaction | 
 6 | 
 18 Adults | 
 APV | 
 SR | 

 
 MASRD [ 117 ] | 
 2017 | 
 Games for Students | 
 0.625 | 
 15 Subjects | 
 V | 
 HA | 

 
 DAiSEE [ 135 ] | 
 2016 | 
 Student Engagement | 
 25 | 
 112 Adults | 
 V | 
 HA | 

 
 
 

### II.1 Datasets for Stress Analysis 

 

#### II.11 Unimodal Datasets

 
 Spontaneous Micro-Gesture (SMG) dataset [ 50 ] presents data on stress and micro-gesture, comprising 821056 frames (8 hours of video) from 40 adults. Mental workload Assessment on n-back task Using wearable Sensor (MAUS) dataset [ 165 ] , collected from 22 adults and Ulm Trier Social Stress Test (TSST) dataset [ 105 ] , with data from 105 participants gathered, focus on stress and mental workload. Stress Recognition in automobile Driver database (DRIVEDB) [ 171 ] , released in 2000 by Healey and Picard at MIT’s Media Lab, pioneered drivers’ stress detection with physiological data from 17 drivers in a lab setting and expert-derived Ground Truth labels. DASPS dataset [ 169 ] contains physiological sensor data from 23 participants to evaluate anxiety.

 
 
 

#### II.12 Multimodal Datasets

 
 The StressID dataset [ 164 ] focuses on multiple stimuli offering 39 hours of audio-physiological-visual data from 65 adults. MuSe-CAR dataset includes 303 recordings from 38 drivers captured using audio-visual and physiological sensors, providing insights into drivers’ stress in real-world scenarios. CLAS dataset [ 168 ] focuses on occupational stress, comprising 31 hours of audio and physiological data from 62 adults, while the WESAD dataset [ 60 ] , collected using questionnaires from 15 adults, contains physiological and motion modality information. VerBIO dataset [ 167 ] delves into bio-behavioral stress with 180 hours of data from 55 students, collected using audio, physiological and thermal sensors across various settings. Passau-SFCH dataset [ 170 ] explores sports-related stress, comprising 11 hours of audio, video and thermal data, capturing insights from footballers in environment.

 
 
 
 

### II.2 Datasets for Depression Analysis 

 

#### II.21 Unimodal Datasets

 
 Sonde Health Free Speech (SH2-FS) dataset [ 91 ] includes recordings of individuals in everyday settings such as cars, homes and workplaces. Its annotations draw upon the Patient Health Questionnaire (PHQ) self-diagnostic test [ 191 ] . Further, the eRisk dataset looks at health risks using over a million sentences from 4427 people. The CLPsych dataset [ 184 ] was constructed using text posts by 1989 subjects while the Pittsburgh dataset [ 188 ] was collected utilizing audio recordings from 49 people. Similarly, the BlackDog dataset [ 183 ] includes audio recordings from 130 participants in open-ended question-answer format. The Chi-Mei Mood disorder database [ 178 ] from the Chi-Mei Medical Center focuses on mood disorders. The UM (University of Maryland) dataset [ 180 ] focuses on depression within an academic context.

 
 
 

#### II.22 Multimodal Datasets

 
 The 2013 [ 186 ] and 2014 [ 185 ] variants of Audio Visual Emotion Challenge (AVEC) pioneered the datasets for depression analysis using audio-visual indicators. Subsequently, the Distress Analysis Interview Corpus - Wizard of Oz (DAIC-WOZ) dataset [ 182 ] was introduced in 2014 and further enhanced to construct the Extended DAIC dataset in 2019 [ 179 ] , both focusing on clinical interviews to assess depression. The RECOLA multimodal database [ 187 ] , developed through interdisciplinary collaboration at Université de Fribourg, integrates physiological, audio and video signals for remote depression assessment. The MPDD dataset [ 172 ] (2025) expands depression research with 228 sessions of audio, textual and visual data. Similarly, the CMDC [ 173 ] and MMDA [ 174 ] datasets capture depression indicators from 167 and 1025 clinical interview sessions, respectively. The EATD dataset [ 175 ] focuses on 486 short Q A interview audios for speech-based screening, while D-Vlog [ 176 ] analyzes 961 vlog recordings from 816 individuals to examine depression markers. The MODMA dataset [ 177 ] contains EEG and audio information from 55 participants, linking neural and vocal features for depression detection.

 
 
 
 

### II.3 Datasets for Engagement Analysis 

 

#### II.31 Unimodal Datasets

 
 The DAiSEE dataset [ 135 ] provides 9068 video clips from 112 users, capturing boredom and engagement in e-learning, while EngageWild [ 160 ] presents 264 videos from 91 subjects across four engagement levels. Expanding on this, EngageNet [ 190 ] includes 31 hours of data from 127 participants with over 11,300 clips, exploring both behavioral and cognitive engagement. Similarly, PAFE [ 49 ] features 15 hours of videos and 1,100 attention probes from 15 students, analyzing attention shifts in online lectures. The VRESEE dataset [ 46 ] extends engagement analysis to 88 Egyptian students with 3,525 recorded videos. Beyond educational contexts, FaceEngage [ 163 ] contains over 700 YouTube gaming videos from 25 amateur gamers, assessing engagement variations based on demographics and gameplay duration. The MASRD dataset [ 117 ] further integrates immersion and presence using the Game Engagement Questionnaire (GEQ). More recently, the DREAMS dataset [ 189 ] introduced 8.7 hours of video from 32 participants, capturing engagement and attention across diverse real-world scenarios.

 
 
 

#### II.32 Multimodal Datasets

 
 The Multimodal Human-Human-Robot Interaction (MHHRI) dataset [ 115 ] analyses interactions between humans and robots. It offers engagement and personality metrics from 18 participants during dyadic and triadic interactions. The User Engagement in Human-Robot Interaction (UE-HRI) dataset [ 118 ] documents spontaneous human interactions with the robot Pepper, focusing on facial expressions and postural details related to engagement.

 
 
 
 

### II.4 Discussion of Datasets 

 
 The aforementioned datasets are crucial for analyzing stress, depression and engagement, yet challenges arise due to differences in modalities, data sizes and labeling techniques. Multimodal datasets like MuSe-CaR , VerBIO , DAIC , RECOLA and MHHRI incorporate diverse data types, making synchronization difficult [ 77 ] . Addressing these challenges requires standardized formats and protocols to enable seamless cross-dataset comparisons and integrative analyses. Notably, to the best of our knowledge, no publicly available dataset jointly annotates any two or all three of these psychological states (stress, depression, engagement), highlighting a key gap in affective computing research. Future datasets should use multi-label annotations to capture overlapping states and refine engagement modeling by distinguishing passive and active engagement, especially in stress or depression contexts.

 
 
 
 

## III Inputs 

 
 Various input modalities used in analyzing stress, depression and engagement are outlined below.

 
 

### III.1 Unimodal Inputs 

 

#### III.11 Visual Modality

 
 The following specific clues are often extracted from images and videos for analysing stress, depression and engagement.

 
 
 
 ■ \blacksquare 
 
 Facial Features : Landmark coordinates, textures and expressions represent physical characteristics, especially in the eyes, nose and mouth regions. They are analyzed to identify emotions like stress and engagement through facial images and videos. He et al. [ 63 ] used dynamic facial appearance for stress analysis, while Li et al. [ 64 ] applied representation learning for depression recognition. In another work, Gupta et al. [ 40 ] used facial emotion recognition in real-time online settings for engagement detection.

 

 ■ \blacksquare 
 
 Action Units (AUs) : AUs describe facial muscle movements linked to emotions and can detect subtle states like stress or suppressed emotions [ 67 ] . To this end, De et al. [ 34 ] explored AU encoding for stress, depression and engagement analysis while Alkabbany et al. [ 68 ] measured engagement by analyzing AUs during learning activities.

 

 ■ \blacksquare 
 
 Eye Tracking Metrics : Gaze, blink rate and saccadic movements help understand focus and attention. They correlate with cognitive engagement [ 70 , 71 ] . In this direction, Savchenko et al. [ 44 ] and Choi et al. [ 72 ] analyzed eye tracking for attention during online learning and video watching.

 

 ■ \blacksquare 
 
 Body Dynamics : Body dynamics, including head movements, posture and body language, convey emotional states through non-verbal cues. For example, Kuttala et al. [ 73 ] explored body dynamics for stress detection, while Alghowinem et al. [ 74 ] used head posture, movements and eye gaze for depression detection.

 

 ■ \blacksquare 
 
 Micro-Gestures : Micro-gestures are subtle, involuntary movements reflecting inner feelings [ 77 ] . In this context, Chen et al. [ 50 ] studied their interplay with emotion states and utilized them for emotional stress analysis.

 

 
 
 
 

#### III.12 Physiological Modality

 
 The frequently used methods to measure physiological responses for stress, depression and engagement analysis are described below. They provide insights about special physiological characteristics associated with stress, depression and engagement [ 104 ] .

 
 
 
 ■ \blacksquare 
 
 Heart Rate Activity : It includes monitoring heart rate variability (HRV) and patterns using Electrocardiogram (ECG) and Photoplethysmogram (PPG) sensors. It is particularly useful for detecting stress and engagement levels, as changes in heart rate can indicate emotional arousal or relaxation. For instance, Giannakakis et al. [ 192 ] utilized HRV for stress analysis. Additionally, remote PPG, which can be classified under both physiological and visual modalities, has been employed in works like those of Sun et al [ 54 ] and Casado et al. [ 109 ] .

 

 ■ \blacksquare 
 
 Electroencephalogram (EEG) : EEG measures brain activity and is often used in the analysis of depression and stress. It provides insight into cognitive processing and emotional regulation. In this context, Xia et al. [ 80 ] and Sharma et al. [ 193 ] used EEG for stress and depression analysis, demonstrating its effectiveness in detecting nuanced changes in mental states.

 

 ■ \blacksquare 
 
 Electrodermal Activity (EDA) : EDA, also known as skin conductance, measures the electrical changes on the skin surface due to sweat gland activity, which is influenced by the sympathetic nervous system. It is a sensitive marker for emotional arousal, stress and engagement. For example, while Zhu et al. [ 79 ] used EDA to detect stress, Alzoubi et al. [ 104 ] reported EDA-based features that help in identifying both depression-related and engagement-related arousal changes.

 

 ■ \blacksquare 
 
 Electromyogram (EMG) : EMG measures the electrical activity produced by skeletal muscles and is indicative of muscle tension, often related to stress or emotional intensity. It’s used for analyzing facial muscle responses in emotional states and stress. In this direction, Pourmohammadi et al. [ 194 ] exemplify the use of EMG along with other biosignals for stress detection.

 

 ■ \blacksquare 
 
 Respiratory Signals : Respiratory rate and breathing patterns are critical indicators of psychological states like stress or relaxation. Changes in breathing can reflect emotional arousal, stress, or engagement levels. For instance, respiratory signals have been used by Shan et al. [ 71 ] and Fernandez et al. [ 78 ] for stress detection.

 

 
 
 
 

#### III.13 Audio Modality

 
 Techniques like prosodic analysis, voice quality and speech rate extract features from audio signals to detect stress, depression and engagement. Huang et al. [ 88 ] used domain adaptation in speech emotion recognition, while Sardari et al. [ 89 ] emphasized audio features for predicting emotions. Suparatpinyo et al. [ 39 ] leveraged acoustic features for stress and depression and Chen et al. [ 35 ] studied intonation and loudness for emotion recognition. Ben et al. [ 118 ] used audio analysis to monitor engagement with the robot Pepper, showing speech cues capture engagement levels. Rejaibi et al. [ 195 ] employed MFCC-based RNN for depression recognition, underscoring speech analysis’s clinical relevance.

 
 
 

#### III.14 Text Modality

 
 Textual data from social media and online platforms also play a pivotal role in identifying stress, depression and engagement. Turcan et al. [ 59 ] examined Reddit discussions to detect stress-related expressions, while Chiong et al. [ 94 ] proposed methods for depressive symptom detection. In educational contexts, Kastrati et al. [ 42 ] showed that text analytics could gauge student engagement by examining discourse cues in online forums. Othmani et al. [ 90 ] employed linguistic features for affect and depression recognition and Lin et al. [ 114 ] introduced SenseMood for depression detection on social media. Together, these works underscore text-based approaches’ adaptability across the three psychological states, offering insight into users’ mental well-being.

 
 
 

#### III.15 Motion Modality

 
 Motion plays a crucial role in understanding and interpreting physical responses to psychological stressors. It is used to analyze bodily movements as indicators of stress levels. Notably, Schmidt et al. [ 60 ] incorporated motion data to establish a foundational approach for stress analysis. Following this, Bobade et al. [ 102 ] enhanced the integration of motion with physiological signals, showing significant improvements in model accuracy and robustness. Similarly, Liu et al. [ 101 ] employed a client-server model that leveraged motion data to refine multimodal representations for stress detection, emphasizing the critical nature of motion data in comprehensive behavioural analyses.

 
 
 Fig. 3 : Sample data inputs and modalities for most commonly used datasets mentioned in Section IV.3 . Along with audio-visual modalities, they use physiological signals such as ECG, EDA and EMG and have labels for valence, arousal, dominance, trustworthiness, depression and engagement categories. 
 
 
 
 

### III.2 Multimodal Inputs 

 
 Real-world analyses of stress, depression and engagement often combine multiple signals such as visual, textual, audio and physiological, for a holistic perspective. For instance:

 
 
 
 ■ \blacksquare 
 
 Audio+Visual (A+V) : Merging speech features like pitch, with facial or gestural cues like micro-expressions uncovers subtle indicators of stress, depression, or engagement [ 115 ] .

 

 ■ \blacksquare 
 
 Audio+Text (A+T) : Adding text to audio improves depression and stress detection, especially with negative self-focus [ 196 ] .

 

 ■ \blacksquare 
 
 Visual+Physiological (V+P) : Pairing facial video with biometrics enhances stress and engagement analysis by linking expression to arousal [ 154 ] .

 

 ■ \blacksquare 
 
 Physiological+Audio (P+A) : Linking biosignals like ECG and EDA to vocal prosody helps validate emotional arousal and detect stress in real-time [ 55 ] .

 

 ■ \blacksquare 
 
 Physiological+Motion (P+M) : Pairing wearable sensor data with accelerometer readings helps capture dynamic changes in stress and engagement [ 60 ] .

 

 ■ \blacksquare 
 
 Audio+Visual+Text (A+V+T) : Incorporating lingual features with audio-visual content enables a holistic analysis by fusing semantics, intonation and appearance [ 115 ] .

 

 ■ \blacksquare 
 
 Audio+Visual+Physiological (A+V+P) : Integrating speech prosody, facial cues and biometric measures enables robust mood disorder detection and engagement monitoring [ 115 ] .

 

 
 
 
 

### III.3 Discussion of Inputs 

 
 Analyzing stress, depression and engagement is challenging due to their complex nature compared to typical emotions [ 4 , 51 ] . Input cues vary by task: subtle micro-gestures [ 77 ] serve as effective visual indicators for high-stress detection, particularly when individuals mask their feelings [ 52 , 54 ] . Physiological signals capture prolonged, subtle changes in stress, while visual modalities detect rapid fluctuations [ 51 , 77 ] . Audio and textual data provide additional emotional context not captured by other modalities [ 91 , 94 ] . Multimodal approaches, combining these inputs, represent the state-of-the-art for analyzing these states [ 74 , 115 ] . Unlike typical emotions with clear cues (e.g., smiling for ‘happy’), these states vary significantly across individuals, making multimodal methods essential for improved understanding and performance [ 197 ] . To effectively capture this variability, selecting fusion techniques tailored to the data’s unique characteristics is crucial for enhancing the accuracy and robustness of computational models [ 198 , 73 ] .

 
 
 
 

## IV Computational Approaches for Stress, Depression and Engagement Analysis 

 
 This section discusses how computational methods analyze stress, depression and engagement by using data from digital interactions, wearables and sensors to create accurate models. These methods, preferred over traditional ones relying on subjective reports and clinical observations, offer a more objective and effective understanding of complex emotional states [ 108 , 23 ] .

 
 
 Fig. 4 : Chronological emergence of computational approaches from traditional ML techniques (e.g., Naive Bayes, Logistic Regression) to advanced DL methods (e.g., GNNs, Transformers) and emerging paradigms (e.g., federated learning, cloud-edge computing), illustrating the evolving complexity and sophistication of stress, depression and engagement analysis. Here, L’ denotes ‘Learning’. 
 
 

### IV.1 Categories of Computational Approaches 

 
 Figure  4 outlines the evolution of computational approaches for analyzing stress, depression and engagement, showing the shift from basic to advanced learning strategies. This section categorizes these approaches into three groups: Machine Learning, Deep Learning and Advanced Learning Approaches. We discuss these paradigms: Supervised learning, using fully labeled data; Unsupervised learning, operating without explicit labels (e.g., anomaly detection); Semi-supervised learning, utilizing a mix of labeled and unlabeled data; and Self-supervised learning, where models derive training signals from the data itself, enhancing efficiency [ 199 , 200 , 201 ] .

 
 

#### IV.11 Traditional Machine Learning

 
 Machine Learning (ML) approaches empower computers to learn from data without requiring explicit rule-based instructions. They often involve hand-engineered features combined with relatively simpler classifiers, particularly in supervised settings. Historically, ML predominated before deep learning gained traction and it remains a strong choice for certain tasks where data availability is limited or model interpretability is crucial [ 79 , 62 ] .

 
 
 In stress analysis, early ML solutions incorporate physiological signals or smartphone sensors to build classical models. For instance, Saugbacs et al.  [ 81 ] relied on accelerometer and gyroscope data to train decision tree and KNN classifiers for stress detection. Likewise, in depression research, De et al.  [ 95 ] introduced a logistic regression framework to diagnose major depressive disorder from social media posts. Classifiers like Naive Bayes, ensemble techniques and random forests have also been employed for multi-modal, small-scale tasks, including depressive state recognition [ 113 ] .

 
 
 Beyond fully supervised pipelines, traditional ML encompasses unsupervised approaches—such as clustering to group stress or depression indicators—and semi-supervised methods, where partially labeled data guides the learning process. Roldan et al.  [ 57 ] , for example, used anomaly detection to flag outlier stress signals in a scenario with minimal labels. Moreover, ML-based engagement analysis typically relies on interpretable features (e.g., facial action units, gaze metrics) to gauge attention and immersion. Taken together, these traditional ML methods still excel when computational resources are constrained, datasets are small, or transparency in decision-making is paramount [ 79 , 62 ] .

 
 
 

#### IV.12 Deep Learning

 
 Deep Learning (DL) is a subset of ML employing neural architectures with multiple layers to automatically learn representations from raw data. Its popularity rose around 2010 (Fig.  4 ). Unlike traditional ML, DL often reduces reliance on hand-crafted features  [ 133 ] . Basic supervised DL techniques such as CNNs and RNNs are extensively applied to single-modality data like facial images or text. Zhou et al.  [ 128 ] used CNNs for depression detection from facial images. Zhong et al.  [ 202 ] leveraged CNN-based discriminant features for robust depression classification. In the textual domain, Orabi et al.  [ 196 ] and Cai et al.  [ 203 ] applied neural networks for depression recognition on social media. Other language-oriented works have used RNNs, BERT and Transformers to analyze depression [ 103 ] , highlighting the growing role of large-scale language models. These neural methods achieve strong performance in stress recognition  [ 40 , 136 ] and engagement prediction  [ 121 , 45 ] .

 
 
 Recent innovations in DL use unsupervised, semi-Supervised and self-Supervised approaches. In this context, Hierarchical attention networks, graph neural networks and Transformers expand beyond simple CNN/RNN architectures  [ 143 , 130 , 138 ] . Fang et al.  [ 108 ] used multi-level attention with limited labeled data (a semi-supervised scenario) to detect depression. Kuttala et al.  [ 73 ] integrated self-supervised hierarchical CNNs for stress analysis. Sun et al.  [ 200 ] introduced an unsupervised/weakly supervised remote physiological measurement approach using spatiotemporal contrast. He et al.  [ 63 ] and Yu et al.  [ 84 ] leveraged weak supervision to detect depression from facial data with minimal labels. In addition, context-aware systems and voice source analysis have also been explored to personalize emotion recognition  [ 162 ] . These strategies reduce annotation costs and often yield robust solutions when labeled data are expensive or scarce.

 
 
 

#### IV.13 Advanced Approaches

 
 Many approaches beyond classical ML or DL used in stress, depression and engagement analysis are discussed below.

 
 
 
 ■ \blacksquare 
 
 Transfer Learning. Transfer Learning reuses models or features from one task or dataset for a new, possibly related, task. This is highly beneficial in mental health contexts. In stress detection, Theerthagiri et al.  [ 58 ] and Albaladejo et al.  [ 204 ] improved classification accuracy by adapting pre-trained networks to new stress datasets. For depression, domain adaptation or advanced voice-recognition models are repurposed, as in  [ 88 , 39 ] . Engagement analysis has similarly seen successful transfer, for example, Khenkar et al.  [ 48 ] leveraged micro-gesture embeddings to interpret learner behavior. Such supervised or semi-supervised transfer solutions accelerate model convergence, handle smaller labeled sets and facilitate domain generalization.

 

 ■ \blacksquare 
 
 Multimodal Learning and Fusion Methods. 
Multimodal learning integrates two or more different data streams (e.g., visual, audio, text, physiological) to obtain a richer representation of psychological states. Chen et al.  [ 119 ] showed the benefit of combining intra- and inter-modal features from IoMT data for depression detection. Xia et al.  [ 120 ] employed a multimodal graph neural network for structured signals in depression detection, while He et al.  [ 27 ] demonstrated how integrating EEG, speech and facial expressions can improve recognition accuracy. Mou et al.  [ 154 ] merged physiological signals with driver behavior for stress detection and Orabi et al.  [ 196 ] combined text plus acoustic data for depression analysis. To systematically fuse these diverse inputs, following fusion strategies are employed.

 
 
 
 – 
 
 Early Fusion (Feature-Level Fusion) : Features from multiple modalities are merged before classification to costruct a comprehensive feature vector that encompasses information from all the modalities. It uses techniques like concatenation, stacking and weighted fusion [ 198 ] .

 

 – 
 
 Late Fusion (Decision-Level Fusion) : Each modality is processed and classified independently. The resulting classifications or scores from each modality are then fused to make a final decision. It is beneficial when each modality provides strong, independent evidence for the classification. Methods like ensemble learning, majority voting and score-level fusion are commonly used. [ 154 ] .

 

 
 

 
 
 
 

#### IV.14 Other Emerging Approaches

 
 Emerging methods are addressing the complexities of real-world mental health analysis through innovative paradigms. Federated Learning [ 36 , 37 ] decentralizes training to preserve privacy, a critical feature for handling sensitive medical data. Contrastive Learning [ 125 ] enhances robustness by learning representations through comparisons of positive and negative sample pairs. Cloud-Edge Computing [ 84 ] optimizes real-time detection of depression and stress by balancing local (on-device) and remote computation. Personalized Learning [ 149 , 134 , 146 ] tailors detection models to individual characteristics, improving accuracy. Blended Learning [ 31 ] integrates traditional and online education, fostering flexible environments that support well-being. Human-Centered Computing and Wearable Technologies are also advancing the field: Chen et al. [ 50 ] utilized body gestures for stress analysis, Shan et al. [ 71 ] developed non-contact respiratory sensors and Lin et al. [ 114 ] combined wearable signals with social media data for depression monitoring. Schmidt et al. [ 60 ] constructed the WESAD dataset for multimodal wearable stress and affect detection. Additionally, secure frameworks like blockchain-based systems [ 52 ] are being explored, highlighting the growing emphasis on robust, privacy-preserving solutions.

 
 
 

#### IV.15 Discussion of Categories of Approaches

 
 The field of stress, depression and engagement analysis has seen significant expansion since 2010 with the adoption of machine ML techniques, further accelerated by advanced DL approaches over the following decade [ 23 ] . This evolution from traditional methods towards ML and DL has not only improved analytical performance [ 116 ] but also shifted the computational landscape towards automated data representation extraction, reducing reliance on manual feature engineering [ 133 ] . Despite these advances, DL technologies come with challenges, including high computational demands, the need for large labeled datasets and complexities in model interpretability [ 95 , 57 ] . Recently, the emphasis on interpretable deep networks has grown, highlighting the importance of transparency and trust in mental health applications [ 36 ] . Moreover, hybrid and multimodal fusion techniques have become popular, balancing performance with computational efficiency [ 40 ] .

 
 
 
 

### IV.2 Computational Analysis Framework 

 

#### IV.21 Generic Phases

 
 Fig. 5 : A depiction of the generic phases used in various computational approaches for stress, depression and engagement analysis. 
 
 
 A general framework for constructing computational models for stress, depression and engagement analysis is depicted in Fig. 5 . Initially, data acquisition and preprocessing eliminate noise and extraneous information. Subsequent transformations involve feature engineering, dimensionality reduction and encoding. The models undergo training and evaluation with advanced methods like K-fold cross-validation [ 205 ] and hyperparameter tuning via grid and random search to enhance performance [ 84 ] . In industrial settings, deployment integrates these models into applications, refining predictions with user feedback. These deployment phases are less emphasized in academic research, which instead prioritizes advancements in concepts, methodologies and algorithms.

 
 
 

#### IV.22 Data Collection and Preprocessing

 
 Data for stress, depression and engagement analysis collected from sources such as video, audio, wearable sensors and online platforms undergo preprocessing before further analysis.

 
 
 
 ■ \blacksquare 
 
 Visual Data Preprocessing :
This involves face detection and tracking across video frames which is crucial for dynamic emotion analysis [ 23 ] . Facial landmark detection is key for precise emotion recognition, pinpointing critical facial expressions [ 105 ] . Noise removal techniques like histogram equalization are also employed for visual data preprocessing [ 109 ] . The face registration and alignment methods are used to standardize the data for consistent facial analysis [ 106 ] .

 

 ■ \blacksquare 
 
 Physiological Signal Preprocessing :
Physiological signals like HRV and EDA provide insight into stress or emotional states. Noise removal in these signals is vital for accurate analysis, typically involving filtering methods to eliminate irrelevant artefacts [ 113 ] . The Z-score-based normalization is also used to improve the comparability across subjects [ 61 ] . Further, dimensionality reduction techniques like PCA help focus on relevant signal aspects while reducing complexity [ 62 ] .

 

 ■ \blacksquare 
 
 Speech Preprocessing :
In speech analysis, extracting features that reflect stress or other emotional states is crucial. Features such as pitch, energy and formant frequencies capture key emotional cues. Normalization (e.g., min-max scaling) adjusts for variations in loudness and speaking rate [ 195 ] . Speech noise removal filters out background disturbances for clearer feature extraction  [ 92 ] . Dimensionality reduction methods like MFCCs distill the speech features.

 

 ■ \blacksquare 
 
 Text Preprocessing :
Textual data analysis involves processing raw text to extract meaningful patterns. Tokenization, stop-word removal and stemming/lemmatization break down the text into analyzable elements [ 57 ] . Normalization like lowercasing and punctuation removal ensures uniformity. Techniques like TF-IDF or word embeddings reduce dimensionality, capturing the text’s semantic essence [ 59 ] .

 

 ■ \blacksquare 
 
 Motion Preprocessing : This step involves noise filtering and sensor calibration for accuracy, alongside principal component analysis to reduce dimensionality and focus on essential motion features [ 101 , 102 ] . Time synchronization and normalization align and scale the data, facilitating its integration with other modalities for comprehensive analysis [ 60 , 206 ] .

 

 
 
 
 

#### IV.23 Model Selection and Training Strategies

 
 Training computational models for stress, depression and engagement analysis presents several challenges [ 133 ] . These include handling small and biased datasets [ 58 ] , ensuring model robustness [ 50 ] , efficiently using multimodal data [ 193 ] and preventing overfitting during model training [ 97 ] . To address these challenges, the following strategies are employed:

 
 
 
 ■ \blacksquare 
 
 Handling Small and Biased Data Sets : Addressing the issues of small and biased datasets is crucial for creating reliable models. Data augmentation methods help by expanding the variety and amount of training data, which reduces overfitting risks [ 207 ] . Similarly, Transfer Learning proves beneficial by applying knowledge from related tasks, enhancing model performance with limited data [ 208 ] .

 

 ■ \blacksquare 
 
 Ensuring Model Robustness : Transfer learning and ensemble models boost model robustness and accuracy. Transfer learning applies insights from related tasks to improve learning, while ensemble methods merge multiple models’ predictions, diversifying decision-making and enhancing emotion and stress recognition [ 90 ] .

 

 ■ \blacksquare 
 
 Utilizing Attention Mechanism for Time-Series Data : The attention mechanism enables the models to focus on crucial sections of time-series data selectively. This technique is particularly beneficial for accurately analyzing stress by examining physiological and behavioural signals, providing a more nuanced understanding of these indicators [ 133 ] .

 

 ■ \blacksquare 
 
 Integrating Multimodal Data for Comprehensive Analysis : Integrating data from various sources, including ECG, EEG and wearable sensors, is crucial for creating detailed models. This approach to fuse complementary information from multiple modalities improves emotion analysis accuracy [ 193 ] .

 

 ■ \blacksquare 
 
 Adopting Enhanced Learning Strategies and Feature Fusion : Using adaptive learning rates and structured feature fusion enhances training efficiency with multi-source data, improving learning speed, consistency and pattern recognition across physiological signals [ 73 ] .

 

 
 
 
 

#### IV.24 Testing and Evaluation Techniques

 
 Models for analyzing stress, depression and engagement are assessed through following methods.

 
 
 
 ■ \blacksquare 
 
 Quantitative Evaluation :
This approach uses numerical metrics such as Accuracy (Acc), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), precision, recall and F1-score to evaluate model performance. Models are automatically evaluated against standard benchmarks for both general and specific analysis, removing the need for manual assessment [ 209 ] .

 

 ■ \blacksquare 
 
 Qualitative Evaluation :
In qualitative assessment, the focus is on descriptively analyzing the model’s output to gauge its capture of emotional context. This sheds light on capabilities and areas for enhancement beyond what quantitative measures reveal. For instance, Chen et al. [ 210 ] analyzed a stress detection model’s reaction to video types and Lin et al. [ 211 ] studied the impact of voice quality on depression recognition, linking pronunciation to predictions.

 

 ■ \blacksquare 
 
 Human Evaluation :
Experts such as psychologists or clinicians conduct evaluations, applying both quantitative and qualitative measures based on their understanding of emotional nuances [ 135 ] . Although this method is more demanding in terms of time and resources compared to automated assessments, it provides an in-depth evaluation, which is especially valuable for identifying stress, depression and engagement states.

 

 
 
 
 
 

### IV.3 State-of-the-Art 

 
 While Section IV.2 presents a generic framework for computational analysis of stress, depression and engagement, this section discusses the latest advancements for the same.

 
 

#### IV.31 State-of-the-Art for Stress Analysis

 
 Stress detection research has expanded rapidly, particularly on the WESAD dataset, where newer methods integrate advanced data augmentation and deep networks to surpass 95% accuracy. For example, Li et al. [ 212 ] employ ConvNeXt and GAN-based augmentation on physiological signals, while Wang et al. [ 213 ] introduce PhysioFormer, achieving 99.54% accuracy through specialized feature extraction. Self-supervised and contrastive techniques are also on the rise: Pulse-PPG [ 214 ] targets PPG signals with over 94% accuracy and COCOA [ 206 ] contrasts multiple sensor modalities. Federated approaches [ 101 ] further secure data privacy by distributing training across client devices without centralizing raw samples. Beyond wearable-only solutions, MuSe-CaR [ 166 ] focuses on driver stress with multimodal inputs (audio-visual, textual), while Siamese Capsule methods [ 215 ] handle continuous stress prediction. This landscape underscores a shift toward more robust, privacy-preserving techniques that fuse multiple signals and leverage powerful augmentation or self-supervision for high-precision stress detection. Table III presents a comprehensive performance evaluation of state-of-the-art methods for stress analysis.

 
 
 

#### IV.32 State-of-the-Art for Depression Analysis

 
 As discussed in Table IV , depression analysis increasingly leverages the AVEC benchmarks, where multimodal techniques capture subtle affective and physiological cues. Dictionary-based methods [ 216 ] apply bidirectional fusion across audio, visual and textual data, reducing mean absolute errors below 4.0. Spatiotemporal fusion [ 217 ] extends these gains by examining continuous facial and physiological patterns. Time-domain speech modeling [ 218 ] refines acoustic features through dual-path attention, boosting detection robustness. In parallel, unsupervised rPPG-based analysis [ 109 ] uncovers remote physiological changes via face videos, yielding a less intrusive approach. Advanced CNN architectures (e.g., MMDepNet [ 219 ] ) combine multiple data streams such as physiological, textual and visual to enhance feature granularity. Collectively, these works show a trend toward deeper neural architectures, multimodal synergy and domain-adaptive learning, effectively lowering errors and improving real-world viability in both clinical and everyday contexts.

 
 
 

#### IV.33 State-of-the-Art for Engagement Analysis

 
 Engagement detection, though less studied than stress and engagement, is gaining traction on datasets like EngageWild [ 160 ] and DAiSEE [ 135 ] . In EngageWild, real-time CNN solutions [ 44 ] utilize EfficientNet or MobileNet variants for on-device inference, while multi-segment LSTM and TCN approaches [ 220 , 33 ] enhance frame-level capture of student affect. On DAiSEE, EngageFormer [ 221 ] integrates physiological and visual data through a transformer design and self-supervised ViT-based autoencoders [ 222 ] tackle facial feature reconstruction for higher accuracy. Other solutions incorporate spatiotemporal networks, like DFSTN [ 65 ] , or ordinal classification [ 223 ] to manage fine-grained engagement levels. Overall, the field is moving toward multimodal fusion and deeper architectures, leveraging flexible sequences or hybrid attention blocks to handle complex behaviors across diverse learning scenarios. Table V presents a comprehensive performance evaluation of state-of-the-art methods for engagement analysis.

 
 
 TABLE III : Performance summary of stress analysis approaches, sorted first by year and then by Accuracy (‘Acc’). Here ‘CCC,’ ‘V,’ ‘P,’ ‘A,’ and ‘M’ denote Concordance Correlation Coefficient, visual, physiological, audio and motion modalities. The abbreviations used include COCOA (Cross Modality Contrastive Learning for Sensor Data), PCA (Principal Component Analysis), ANN (Artificial Neural Network), RF (Random Forest), DT (Decision Tree), GAN (Generative Adversarial Network) and SVM (Support Vector Machine). 
 
 
 
 
 
 Dataset 
 | 
 
 
 Method 
 | 
 
 
 Year 
 | 
 
 
 Architecture 
 | 
 
 
 Modality 
 | 
 
 
 CCC 
 | 
 
 
 Acc 
 | 

 
 Stress | 
 | 

 
 
 
 
 
 WESAD [ 60 ] 
 
 | 
 
 
 Wearables’ feature analysis [ 224 ] 
 | 
 
 
 2025 
 | 
 
 
 XGBoost + DT + Transfer learning 
 | 
 
 
 P 
 | 
 
 
 – 
 | 
 
 
 99.00% 
 | 

 
 | 
 
 
 Pulse-PPG [ 214 ] 
 | 
 
 
 2025 
 | 
 
 
 Contrastive learning-based model 
 | 
 
 
 P 
 | 
 
 
 – 
 | 
 
 
 94.52% 
 | 

 
 | 
 
 
 PhysioFormer [ 213 ] 
 | 
 
 
 2024 
 | 
 
 
 ContribNet + AffectNet 
 | 
 
 
 P 
 | 
 
 
 – 
 | 
 
 
 99.54% 
 | 

 
 | 
 
 
 Data augmentation [ 212 ] 
 | 
 
 
 2024 
 | 
 
 
 ConvNeXt + Self-attention GAN 
 | 
 
 
 P 
 | 
 
 
 – 
 | 
 
 
 95.70% 
 | 

 
 | 
 
 
 Self-supervised learning [ 225 ] 
 | 
 
 
 2023 
 | 
 
 
 Temporal convolution + Transformer 
 | 
 
 
 P 
 | 
 
 
 – 
 | 
 
 
 96.29% 
 | 

 
 | 
 
 
 Cross-modality contrastive learning [ 206 ] 
 | 
 
 
 2022 
 | 
 
 
 COCOA 
 | 
 
 
 PM 
 | 
 
 
 – 
 | 
 
 
 97.60% 
 | 

 
 | 
 
 
 Multimodal representation [ 101 ] 
 | 
 
 
 2021 
 | 
 
 
 Client-server aggregated model 
 | 
 
 
 PM 
 | 
 
 
 – 
 | 
 
 
 93.20% 
 | 

 
 | 
 
 
 Self-supervised learning [ 201 ] 
 | 
 
 
 2020 
 | 
 
 
 Multi-task CNN 
 | 
 
 
 P 
 | 
 
 
 – 
 | 
 
 
 96.90% 
 | 

 
 | 
 
 
 Multimodal bio-signal analysis [ 102 ] 
 | 
 
 
 2020 
 | 
 
 
 PCA + ANN + RF 
 | 
 
 
 PM 
 | 
 
 
 – 
 | 
 
 
 95.21% 
 | 

 
 | 
 
 
 Multimodal fusion [ 226 ] 
 | 
 
 
 2020 
 | 
 
 
 Bimodal Deep AutoEncoder 
 | 
 
 
 P 
 | 
 
 
 – 
 | 
 
 
 90.25% 
 | 

 
 | 
 
 
 Base paper (two-class) [ 60 ] 
 | 
 
 
 2018 
 | 
 
 
 SVM + ANN + RF 
 | 
 
 
 PM 
 | 
 
 
 – 
 | 
 
 
 93.12% 
 | 

 
 | 
 
 
 Multimodal bio-signals (three-class) [ 102 ] 
 | 
 
 
 2018 
 | 
 
 
 PCA + ANN + RF 
 | 
 
 
 PM 
 | 
 
 
 – 
 | 
 
 
 84.32% 
 | 

 
 | 
 
 
 Base paper (three-class) [ 60 ] 
 | 
 
 
 2018 
 | 
 
 
 SVM + ANN + RF 
 | 
 
 
 PM 
 | 
 
 
 – 
 | 
 
 
 80.34% 
 | 

 
 
 
 
 
 MuSe-CaR [ 166 ] 
 
 | 
 
 
 SCapsNet [ 215 ] 
 | 
 
 
 2024 
 | 
 
 
 Siamese + Capsule Net + Optimization 
 | 
 
 
 AVT 
 | 
 
 
 0.5072 
 | 
 
 
 – 
 | 

 
 | 
 
 
 DeepSpectrum [ 106 ] 
 | 
 
 
 2022 
 | 
 
 
 Early fusion + Attention 
 | 
 
 
 APVT 
 | 
 
 
 0.4585 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Multitask learning [ 227 ] 
 | 
 
 
 2021 
 | 
 
 
 Self-attention + Bi-LSTM + EfficientNet 
 | 
 
 
 AV 
 | 
 
 
 0.3587 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Attention enhanced recurrent net [ 228 ] 
 | 
 
 
 2021 
 | 
 
 
 VGGFace + DeBERTa + Attention 
 | 
 
 
 ATV 
 | 
 
 
 0.3803 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Hybrid (early + late) fusion [ 106 ] 
 | 
 
 
 2021 
 | 
 
 
 VGGish + VGGface + OpenFace + BERT 
 | 
 
 
 APVT 
 | 
 
 
 0.4646 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Aligned Annotation Weighting [ 105 ] 
 | 
 
 
 2021 
 | 
 
 
 LSTM + RNN 
 | 
 
 
 APVT 
 | 
 
 
 0.4913 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Multimodal sentiment analysis [ 166 ] 
 | 
 
 
 2021 
 | 
 
 
 BERT + FastText + VGGish 
 | 
 
 
 ATV 
 | 
 
 
 0.5384 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Attention enhanced recurrent net [ 228 ] 
 | 
 
 
 2021 
 | 
 
 
 Wav2vec + DeBERT + Attention 
 | 
 
 
 AV 
 | 
 
 
 0.5558 
 | 
 
 
 – 
 | 

 
 
 
 TABLE IV : Performance summary of depression analysis approaches, sorted first by year and then by mean absolute error (‘MAE’). Here, ‘RMSE,’ ‘V,’ ‘P,’ ‘A,’ and ‘T’ denote root mean square error, visual, physiological, audio and textual modalities. The acronyms used include SSD (Single Shot Multibox Detection network), RFR (Random Forest Regressor), LGBPTOP (Local dynamic appearance descriptor), LPQ (Local Phase Quantisation), C3D/3DCNN (3D CNN), 2DCNN (2D CNN), FDHH (Feature Dynamic History Histogram), SVR (Support Vector Regression), LPQ (Local Phase Quantization) and TOP (Temporal Occurrence Pattern). 
 
 
 
 
 
 Dataset 
 | 
 
 
 Method 
 | 
 
 
 Year 
 | 
 
 
 Architecture 
 | 
 
 
 Modality 
 | 
 
 
 MAE 
 | 
 
 
 RMSE 
 | 

 
 
 
 
 
 AVEC 2013 [ 186 ] 
 
 | 
 
 
 Dictionary‐based decomposition [ 216 ] 
 | 
 
 
 2025 
 | 
 
 
 Bidirectional multimodal fusion 
 | 
 
 
 ATV 
 | 
 
 
 3.87 
 | 
 
 
 5.21 
 | 

 
 | 
 
 
 Long-term spatio-temporal routing [ 217 ] 
 | 
 
 
 2025 
 | 
 
 
 Spatiotemporal fusion ensemble network 
 | 
 
 
 PV 
 | 
 
 
 5.38 
 | 
 
 
 6.74 
 | 

 
 | 
 
 
 Time-domain speech modeling [ 218 ] 
 | 
 
 
 2025 
 | 
 
 
 Dual-path state-space attention network 
 | 
 
 
 A 
 | 
 
 
 8.35 
 | 
 
 
 9.05 
 | 

 
 | 
 
 
 Multimodal Depression analysis [ 219 ] 
 | 
 
 
 2024 
 | 
 
 
 MMDepNet 
 | 
 
 
 APTV 
 | 
 
 
 6.15 
 | 
 
 
 7.89 
 | 

 
 | 
 
 
 Unsupervised rPPG-based analysis [ 109 ] 
 | 
 
 
 2023 
 | 
 
 
 SSD + LGBPTOP + RFR + ResNet-50 
 | 
 
 
 PV 
 | 
 
 
 6.43 
 | 
 
 
 8.01 
 | 

 
 | 
 
 
 Depth-wise convolution analysis [ 229 ] 
 | 
 
 
 2021 
 | 
 
 
 3DCNN + SVR 
 | 
 
 
 AV 
 | 
 
 
 6.19 
 | 
 
 
 8.02 
 | 

 
 | 
 
 
 Two-stream image analysis [ 34 ] 
 | 
 
 
 2020 
 | 
 
 
 Two-stream 2DCNN 
 | 
 
 
 V 
 | 
 
 
 5.96 
 | 
 
 
 7.97 
 | 

 
 | 
 
 
 Deep residual learning [ 129 ] 
 | 
 
 
 2019 
 | 
 
 
 ResNet-50 
 | 
 
 
 V 
 | 
 
 
 6.30 
 | 
 
 
 8.25 
 | 

 
 | 
 
 
 Multi-channel ensembling [ 128 ] 
 | 
 
 
 2018 
 | 
 
 
 Four DCNNs 
 | 
 
 
 V 
 | 
 
 
 6.20 
 | 
 
 
 8.28 
 | 

 
 | 
 
 
 Depth-wise video analysis [ 230 ] 
 | 
 
 
 2018 
 | 
 
 
 C3D 
 | 
 
 
 V 
 | 
 
 
 7.37 
 | 
 
 
 9.28 
 | 

 
 | 
 
 
 Dual-channel analysis [ 231 ] 
 | 
 
 
 2017 
 | 
 
 
 Two DCNN 
 | 
 
 
 V 
 | 
 
 
 7.58 
 | 
 
 
 9.82 
 | 

 
 | 
 
 
 Local pattern analysis [ 232 ] 
 | 
 
 
 2015 
 | 
 
 
 LPQ-TOP + MFA 
 | 
 
 
 V 
 | 
 
 
 8.22 
 | 
 
 
 10.27 
 | 

 
 | 
 
 
 Eye-based feature extraction [ 233 ] 
 | 
 
 
 2014 
 | 
 
 
 LPQ + Geo 
 | 
 
 
 AV 
 | 
 
 
 7.86 
 | 
 
 
 9.72 
 | 

 
 | 
 
 
 Baseline paper [ 186 ] 
 | 
 
 
 2013 
 | 
 
 
 OpenSMILE + LGBP-TOP 
 | 
 
 
 AV 
 | 
 
 
 10.88 
 | 
 
 
 13.61 
 | 

 
 
 
 
 
 AVEC 2014 [ 185 ] 
 
 | 
 
 
 Dictionary‐based decomposition [ 216 ] 
 | 
 
 
 2025 
 | 
 
 
 Bidirectional multimodal fusion 
 | 
 
 
 ATV 
 | 
 
 
 3.63 
 | 
 
 
 5.05 
 | 

 
 | 
 
 
 Long-term spatio-temporal routing [ 217 ] 
 | 
 
 
 2025 
 | 
 
 
 Spatiotemporal fusion ensemble network 
 | 
 
 
 PV 
 | 
 
 
 5.09 
 | 
 
 
 6.83 
 | 

 
 | 
 
 
 Time-domain speech modeling [ 218 ] 
 | 
 
 
 2025 
 | 
 
 
 Dual-path state-space attention network 
 | 
 
 
 A 
 | 
 
 
 8.39 
 | 
 
 
 9.14 
 | 

 
 | 
 
 
 Multimodal Depression Analysis [ 219 ] 
 | 
 
 
 2024 
 | 
 
 
 MMDepNet 
 | 
 
 
 APTV 
 | 
 
 
 6.14 
 | 
 
 
 8.11 
 | 

 
 | 
 
 
 Unsupervised rPPG-based analysis [ 109 ] 
 | 
 
 
 2023 
 | 
 
 
 SSD + LGBPTOP + RFR + ResNet-50 
 | 
 
 
 PV 
 | 
 
 
 6.57 
 | 
 
 
 8.49 
 | 

 
 | 
 
 
 Depth-wise convolutional analysis [ 229 ] 
 | 
 
 
 2021 
 | 
 
 
 3DCNN + SVR 
 | 
 
 
 AV 
 | 
 
 
 6.14 
 | 
 
 
 7.98 
 | 

 
 | 
 
 
 Two-stream image analysis [ 34 ] 
 | 
 
 
 2020 
 | 
 
 
 Two-stream 2DCNN 
 | 
 
 
 V 
 | 
 
 
 6.20 
 | 
 
 
 7.94 
 | 

 
 | 
 
 
 Deep residual learning [ 129 ] 
 | 
 
 
 2019 
 | 
 
 
 ResNet-50 
 | 
 
 
 V 
 | 
 
 
 6.15 
 | 
 
 
 8.23 
 | 

 
 | 
 
 
 Multi-channel ensembling [ 128 ] 
 | 
 
 
 2018 
 | 
 
 
 Four DCNN 
 | 
 
 
 V 
 | 
 
 
 6.21 
 | 
 
 
 8.39 
 | 

 
 | 
 
 
 Depth-wise video analysis [ 230 ] 
 | 
 
 
 2018 
 | 
 
 
 C3D 
 | 
 
 
 V 
 | 
 
 
 7.22 
 | 
 
 
 9.20 
 | 

 
 | 
 
 
 Dual-channel analysis [ 231 ] 
 | 
 
 
 2017 
 | 
 
 
 Two DCNN 
 | 
 
 
 V 
 | 
 
 
 7.47 
 | 
 
 
 9.55 
 | 

 
 | 
 
 
 Feature embedding based network [ 159 ] 
 | 
 
 
 2017 
 | 
 
 
 VGG + FDHH 
 | 
 
 
 V 
 | 
 
 
 6.68 
 | 
 
 
 8.04 
 | 

 
 | 
 
 
 Ensembled feature extraction [ 234 ] 
 | 
 
 
 2014 
 | 
 
 
 LGBP-TOP + LPQ 
 | 
 
 
 AV 
 | 
 
 
 8.20 
 | 
 
 
 10.27 
 | 

 
 | 
 
 
 Base paper [ 185 ] 
 | 
 
 
 2014 
 | 
 
 
 OpenSMILE + LPQ 
 | 
 
 
 AV 
 | 
 
 
 8.86 
 | 
 
 
 10.86 
 | 

 
 
 
 TABLE V : Performance summary of engagement analysis approaches, sorted first by year and then by mean absolute error (‘MAE’). Here, ‘Acc’ denotes accuracy and the acronyms used include TCN (Temporal Convolutional Network), LSTM (Long Short-Term Memory), GAP (Gaze-AU-Pose), LBP-TOP (Local Binary Patterns from Three Orthogonal Planes), Deep Facial Spatiotemporal Network (DFSTN), S-WL (Sampling and weighted loss) and LRCN (Long-Term Recurrent Convolutional Network). 
 
 
 
 
 
 Dataset 
 | 
 
 
 Method 
 | 
 
 
 Year 
 | 
 
 
 Architecture 
 | 
 
 
 Modality 
 | 
 
 
 MAE 
 | 
 
 
 Acc 
 | 

 
 
 
 
 
 EngageWild [ 160 ] 
 
 | 
 
 
 Facial feature fusion [ 235 ] 
 | 
 
 
 2023 
 | 
 
 
 Transformer based fusion encoder 
 | 
 
 
 V 
 | 
 
 
 0.0820 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Sequence embedding optimization [ 47 ] 
 | 
 
 
 2022 
 | 
 
 
 Multi-task training 
 | 
 
 
 V 
 | 
 
 
 0.0427 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Real-time CNN classification [ 44 ] 
 | 
 
 
 2022 
 | 
 
 
 EfficientNet-B0 + Ridge regression 
 | 
 
 
 V 
 | 
 
 
 0.0563 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Real-time CNN classification [ 44 ] 
 | 
 
 
 2022 
 | 
 
 
 EfficientNet-B2 + Ridge regression 
 | 
 
 
 V 
 | 
 
 
 0.0702 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Real-time CNN classification [ 44 ] 
 | 
 
 
 2022 
 | 
 
 
 MobileNet + Ridge regression 
 | 
 
 
 V 
 | 
 
 
 0.0722 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Affective ordinal classification [ 33 ] 
 | 
 
 
 2021 
 | 
 
 
 Clip-level features + TCN 
 | 
 
 
 V 
 | 
 
 
 0.0508 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Attention-based hybrid model [ 103 ] 
 | 
 
 
 2020 
 | 
 
 
 Attention-based GRU hybrid net 
 | 
 
 
 V 
 | 
 
 
 0.0517 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Attention-based hybrid model [ 103 ] 
 | 
 
 
 2020 
 | 
 
 
 VGG 
 | 
 
 
 V 
 | 
 
 
 0.0653 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Multi-segment feature analysis [ 220 ] 
 | 
 
 
 2019 
 | 
 
 
 LSTM + FC layers 
 | 
 
 
 V 
 | 
 
 
 0.0572 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Bootstrap ensemble learning [ 236 ] 
 | 
 
 
 2019 
 | 
 
 
 OpenPose + LSTM 
 | 
 
 
 V 
 | 
 
 
 0.0717 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Cluster-attention ensemble [ 76 ] 
 | 
 
 
 2018 
 | 
 
 
 Attention-based NN 
 | 
 
 
 V 
 | 
 
 
 0.0441 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Sequential behaviour analysis [ 237 ] 
 | 
 
 
 2018 
 | 
 
 
 GAP + LBP-TOP 
 | 
 
 
 V 
 | 
 
 
 0.0569 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Segment-level feature analysis [ 238 ] 
 | 
 
 
 2018 
 | 
 
 
 Dilated-TCN 
 | 
 
 
 V 
 | 
 
 
 0.0655 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Feature learning [ 237 ] 
 | 
 
 
 2018 
 | 
 
 
 Gaze-AU-Pose - GAP 
 | 
 
 
 V 
 | 
 
 
 0.0671 
 | 
 
 
 – 
 | 

 
 | 
 
 
 Base paper [ 160 ] 
 | 
 
 
 2018 
 | 
 
 
 OpenFace + LSTM 
 | 
 
 
 V 
 | 
 
 
 0.1000 
 | 
 
 
 – 
 | 

 
 
 
 
 
 DAiSEE [ 135 ] 
 
 | 
 
 
 EngageFormer [ 221 ] 
 | 
 
 
 2025 
 | 
 
 
 Multi-view transformer 
 | 
 
 
 PV 
 | 
 
 
 – 
 | 
 
 
 63.90% 
 | 

 
 | 
 
 
 Self-supervised masked autoencoder [ 222 ] 
 | 
 
 
 2024 
 | 
 
 
 ViT-based facial autoencoder 
 | 
 
 
 PV 
 | 
 
 
 – 
 | 
 
 
 64.74% 
 | 

 
 | 
 
 
 Multimodal engagement detection [ 239 ] 
 | 
 
 
 2024 
 | 
 
 
 VisioPhysioENet 
 | 
 
 
 PV 
 | 
 
 
 – 
 | 
 
 
 63.09% 
 | 

 
 | 
 
 
 Temporal recognition network [ 46 ] 
 | 
 
 
 2022 
 | 
 
 
 EfficientNet B7 + LSTM 
 | 
 
 
 V 
 | 
 
 
 – 
 | 
 
 
 67.48% 
 | 

 
 | 
 
 
 Temporal recognition network [ 46 ] 
 | 
 
 
 2022 
 | 
 
 
 EfficientNet B7 + Bi-LSTM 
 | 
 
 
 V 
 | 
 
 
 – 
 | 
 
 
 66.39% 
 | 

 
 | 
 
 
 Temporal recognition network [ 46 ] 
 | 
 
 
 2022 
 | 
 
 
 EfficientNet B7 + TCN 
 | 
 
 
 V 
 | 
 
 
 – 
 | 
 
 
 64.67% 
 | 

 
 | 
 
 
 Affective ordinal classification [ 33 ] 
 | 
 
 
 2021 
 | 
 
 
 Ordinal TCN 
 | 
 
 
 V 
 | 
 
 
 – 
 | 
 
 
 67.40% 
 | 

 
 | 
 
 
 Dynamic engagement classifier [ 223 ] 
 | 
 
 
 2021 
 | 
 
 
 ResNet + TCN 
 | 
 
 
 V 
 | 
 
 
 – 
 | 
 
 
 63.90% 
 | 

 
 | 
 
 
 Dynamic engagement classifier [ 223 ] 
 | 
 
 
 2021 
 | 
 
 
 ResNet + LSTM 
 | 
 
 
 V 
 | 
 
 
 – 
 | 
 
 
 61.15% 
 | 

 
 | 
 
 
 Dynamic engagement classifier [ 223 ] 
 | 
 
 
 2021 
 | 
 
 
 C3D + TCN 
 | 
 
 
 V 
 | 
 
 
 – 
 | 
 
 
 59.97% 
 | 

 
 | 
 
 
 Spatiotemporal engagement detector [ 65 ] 
 | 
 
 
 2021 
 | 
 
 
 DFSTN 
 | 
 
 
 V 
 | 
 
 
 – 
 | 
 
 
 58.84% 
 | 

 
 | 
 
 
 Dynamic engagement classifier [ 223 ] 
 | 
 
 
 2021 
 | 
 
 
 ResNet + TCN (S-WL) 
 | 
 
 
 V 
 | 
 
 
 – 
 | 
 
 
 53.70% 
 | 

 
 | 
 
 
 Class balanced I3D Model [ 240 ] 
 | 
 
 
 2019 
 | 
 
 
 Inflated 3D CNN 
 | 
 
 
 V 
 | 
 
 
 – 
 | 
 
 
 52.35% 
 | 

 
 | 
 
 
 Base paper [ 135 ] 
 | 
 
 
 2016 
 | 
 
 
 C3D LRCN 
 | 
 
 
 V 
 | 
 
 
 – 
 | 
 
 
 57.90% 
 | 

 
 
 
 

#### IV.34 Discussion of State-of-the-Art

 
 Recent advancements in the fields of stress, depression and engagement analysis have increasingly leveraged deep multimodal fusion techniques to accurately capture subtle behavioral and emotional cues. In stress detection using the WESAD dataset, innovative methods using ConvNeXt and GAN based augmentations have achieved high accuracies, while models such as PhysioFormer have demonstrated exceptional performance through advanced feature extraction techniques [ 212 , 213 ] . For depression analysis on the AVEC benchmarks, techniques that integrate bidirectional fusion with sophisticated spatiotemporal approaches have refined feature detection and significantly reduced error rates [ 216 , 217 ] . Recent trends in depression analysis indicate notable improvements in error reduction, underscoring the effectiveness of advanced multimodal fusion strategies in capturing subtle depressive cues. In engagement analysis, the EngageWild dataset and DAiSEE have long served as widely used resources; however, following the introduction of EngageNet in 2023 [ 190 ] , which offers enhanced annotation precision and improved experimental robustness, the community has increasingly shifted toward adopting EngageNet over EngageWild, with transformer based models on DAiSEE further expanding the methodological landscape [ 221 ] .

 
 
 
 
 

## V Applications of Stress, Depression and Engagement Analysis 

 
 The applications of stress, depression and engagement analysis in mental health and other areas are described in Fig. 6 and below.

 
 

### V.1 Mental Health Applications 

 

#### V.11 Workplace and Occupational Well-being

 
 
 ■ \blacksquare 
 
 Transport and Drivers’ Safety : By analyzing signals such as heart rate and driving behaviour, researchers have developed methods to alert drivers about their stress levels [ 55 , 24 ] . These systems detect stress in real-time, offering suggestions like adjusting cabin settings or applying brakes to enhance safety [ 241 , 154 ] , thereby improving vehicle control during stressful situations [ 242 ] .

 

 ■ \blacksquare 
 
 Health Professionals’ Well-being : Computational tools are used to understand the ways in which workplace dynamics can influence employee engagement [ 243 ] . This analysis aids in the development of enhanced mental health strategies tailored for unusual or atypical situations. For example, researchers have conducted detailed studies on the impact of stress stemming from remote work during the COVID-19 crisis [ 155 ] .

 

 ■ \blacksquare 
 
 Social Workers’ Mental Health : Research indicates that high job demands often result in stress and burnout among social workers [ 156 ] . It has also linked these demands to their engagement and mental health, guiding the creation of strategies to help them manage stress [ 157 ] .

 

 ■ \blacksquare 
 
 Online Meetings: The significance of computational analysis in online meetings is on the rise within the remote work environment [ 31 ] . In this context, the analysis of webcam videos provides instant feedback on attentiveness, aiding in the adaptation of meeting methods [ 49 ] . Additionally, stress during online meetings has been analyzed using remote physiological signals and behavioural features [ 54 ] .

 

 
 
 
 

#### V.12 Detecting Mental Health Disorders

 
 
 ■ \blacksquare 
 
 Anxiety and Stress Detection : Computational methods are increasingly being applied to detect and evaluate mental health conditions. Video-based facial analysis has been used to assess anxiety symptoms [ 205 ] . Additionally, wearable technology has been used to measure physiological responses to stress [ 148 , 85 ] , while ECG and EMG signals have been analyzed for stress detection.

 

 ■ \blacksquare 
 
 Depression Screening and Suicide Prevention : Computational analysis has enabled early detection of depressive and suicidal tendencies through monitoring social media, which aids in timely interventions [ 97 ] . It also supports suicide prevention by identifying risk factors in adolescents engaged with online programs [ 142 ] and suggests that measuring life satisfaction may predict late-life depression and suicide attempts [ 158 ] .

 

 ■ \blacksquare 
 
 Detection of Post-Traumatic Stress Disorder (PTSD) : Innovative tools for detecting engagement during vagal nerve stimulation therapy show promise for treating stress from traumatic events [ 15 , 244 , 245 ] . Additionally, examining social connections post-stroke reveals their impact on depression and physical impairment, suggesting therapeutic strategies [ 246 ] .

 

 
 
 
 

#### V.13 Health and behaviour Monitoring

 
 
 ■ \blacksquare 
 
 Elderly Care : Stress, depression and engagement analysis techniques facilitate continuous monitoring of the elderly’s physical and emotional states. They have been used to aid in distress and disease prediction [ 24 , 159 ] . Additionally, wearable technology is used to improve cognitive training in older adults [ 53 ] .

 

 ■ \blacksquare 
 
 Infant Monitoring : The advancements in Child–Robot Interaction demonstrate that robots, like the ‘Mio Amico’ robot, can adapt to children’s engagement levels [ 247 ] . They can continuously monitor infants and suggest urgent actions.

 

 ■ \blacksquare 
 
 Addiction Monitoring : The computational techniques for stress and engagement analysis facilitate the monitoring of addiction-related behaviours [ 40 ] . For instance, analyzing behavioural responses during food consumption offers insights into addiction and emotional eating patterns [ 109 ] .

 

 
 
 
 Fig. 6 : Mental health and other applications of the computational analysis of stress, depression and engagement. These images were created with DALL·E 2 . 
 
 
 

#### V.14 Treatment Planning for Mental Health Issues

 
 
 ■ \blacksquare 
 
 Mental Health Assessment : Emotion assessment, essential to mental health treatment planning, is supported by facial expression analysis and emotion recognition [ 41 ] . It utilizes clinical interviews, behavioural analysis and social media analysis approaches [ 143 ] .

 

 ■ \blacksquare 
 
 Therapeutic Interventions and Counseling : Understanding stress, depression and engagement provides valuable real-time feedback, thereby improving the effectiveness of therapeutic interventions and counselling sessions [ 61 , 154 ] .

 

 ■ \blacksquare 
 
 Personalized Coping Mechanisms : To aid individuals experiencing challenging emotional states, personalized coping strategy systems have been developed [ 79 , 111 ] . These systems offer tailored support to users facing various forms of emotional distress.

 

 ■ \blacksquare 
 
 Treating Cognitive Degeneration Disorders : Computational methods and wearable sensors help older adults with cognitive training and stress detection [ 53 , 51 ] . These methods use small sensors to track the heart rate and body movement. By analyzing the data from these sensors, researchers can monitor the cognitive load and stress levels to facilitate training for older adults.

 

 
 
 
 

#### V.15 Implementation Solutions for Mental Health

 
 
 ■ \blacksquare 
 
 Mobile Application Development : Computational analysis has significantly contributed to personalized mental healthcare via mobile applications. These applications utilize user interaction data, such as typing speed and phone usage, to non-invasively assess stress levels, enabling real-time monitoring and intervention for stress-related conditions [ 131 , 28 ] .

 

 ■ \blacksquare 
 
 Wearable Technology-based Well-being Analysis : Wearable devices have revolutionized real-time well-being analysis [ 60 ] . Privacy-preserving stress monitoring is now possible with smartwatches [ 37 ] , while wrist devices for electrodermal activity can be used for non-invasive stress detection [ 148 ] .

 

 
 
 
 
 

### V.2 Other Applications 

 

#### V.21 Education and Learning Analytics

 
 
 ■ \blacksquare 
 
 Improving Student Engagement : Computational analysis has improved online learning by detecting student engagement using facial emotion recognition [ 70 , 40 ] . Techniques have been developed to predict mind-wandering during online lectures and to detect students’ engagement in classroom environments [ 49 , 134 ] .

 

 ■ \blacksquare 
 
 Personalized Learning : Adaptive learning technologies have been developed to improve personalized experiences [ 162 ] . They also enable customized support by utilizing automated processes to meet the unique needs of each learner [ 161 ] .

 

 
 
 
 

#### V.22 Gaming and Entertainment

 
 
 ■ \blacksquare 
 
 Creating More Engaging Gaming Experiences : The player engagement in game-based learning can be explored by measuring physiological signals [ 83 ] . Games have also been used as therapeutic interventions for depression [ 108 , 107 ] .

 

 ■ \blacksquare 
 
 Improving the Virtual Reality (VR) Experience : To enhance work engagement and alleviate stress levels, researchers have developed and implemented virtual reality-based games [ 153 ] . These immersive experiences leverage advanced technology to create interactive environments, offering employees an engaging and stress-relieving alternative to traditional methods.

 

 
 
 
 

#### V.23 Human-Computer Interaction (HCI)

 
 
 ■ \blacksquare 
 
 Engagement Detection in HCI : Advanced DL techniques have been employed to measure users’ engagement in HCI scenarios using video-based facial expressions and consumer interaction patterns on social media platforms [ 33 ] .

 

 ■ \blacksquare 
 
 Understanding Human Emotions in HCI : Advancements in HCI have led to applications such as creating emotion databases, using neural networks for emotion detection and exploring settings like classrooms and gaming to monitor and respond to users’ emotional states [ 152 ] .

 

 ■ \blacksquare 
 
 Digital Engagement and Social Media Analysis : By analyzing online behaviour patterns, researchers can identify social trends [ 42 ] . Digital engagement analysis can be used to understand public sentiments and mental health aspects [ 98 ] .

 

 
 
 
 

#### V.24 Ethics and Privacy Preservation

 
 Given the heightened risk of data breaches, computational tools are being developed to ethically handle mental health data, enhancing well-being while protecting privacy and rights [ 95 ] .

 
 
 

#### V.25 Policy Making and Social Support

 
 The insights from stress, depression and engagement analysis are useful in developing strategies to address mental health concerns [ 75 ] . These insights are instrumental for governments and policymakers in devising effective social support initiatives [ 142 ] .

 
 
 
 

### V.3 Discussion of Applications 

 
 In applying computational analysis to mental health and related fields, the context dependency of engagement, stress and depression is crucial. Identifying whether an individual is engaged with specific content or a particular person is often more important than assessing engagement alone. For instance, in older adults with dementia, engagement with either a recommender system or a human partner can promote cognitive activation [ 248 ] , while in human-robot collaborative learning environments, distinguishing whether a learner’s attention is directed toward the robot or a human instructor is essential for effective adaptivity [ 147 ] . Since heightened stress and depression reduce motivation and focus, interventions increasingly target these states, particularly in workplaces and education, to sustain engagement [ 20 , 16 ] . These examples highlight the need for nuanced, context-aware approaches that optimize user experience and therapeutic efficacy.

 
 
 
 

## VI Challenges and Future Directions 

 
 Computational analysis of stress, depression and engagement uncovers the following challenges and future research avenues.

 
 
 
 ■ \blacksquare 
 
 Emerging Generative AI Approaches :
Advancements in generative AI, such as leveraging large language models (LLMs) or generating synthetic data, have shown promise in various domains. Yet, their application in analyzing stress, depression and engagement is less explored. Recent efforts include generating synthetic health sensor data for stress detection [ 132 ] , enhancing educational engagement through generative tools [ 150 ] and assessing LLM capabilities for depression detection [ 145 ] . Future research should employ generative AI techniques to further mental health analysis.

 

 ■ \blacksquare 
 
 Lack of Large-scale Datasets :
The analysis of stress, depression and engagement is hindered by the scarcity of large-scale datasets. Collaborative efforts across disciplines can address this challenge by pooling resources, sharing data and conducting joint analyses within privacy guidelines. Such collaboration enhances dataset quality and availability [ 49 ] .

 

 ■ \blacksquare 
 
 Data Labeling and Multimodal Data Integration :
With the rise of wearable devices and IoT, there are numerous new data sources available. While these offer rich insights, labelling them effectively remains a challenge [ 45 ] . The research efforts are required to create robust labelling methodologies and integrate the diverse data streams to form a coherent understanding of mental states [ 43 ] .

 

 ■ \blacksquare 
 
 Model Generalization :
The emergence of stress, depression and engagement varies greatly among individuals, presenting challenges for generalizing computational models [ 99 ] . Techniques like domain adaptation and multi-task learning offer potential solutions by transferring knowledge between datasets to account for variability in emotional analysis [ 88 ] . Future efforts should focus on refining these models for improved prediction and response to emotional shifts [ 117 ] .

 

 ■ \blacksquare 
 
 Dynamic Analysis :
Emotions are dynamic and subject to change over time. Some emotions can shift very quickly, while others may change more slowly [ 11 ] . This variability presents a challenge for computational models, as they must be capable of adapting to and accurately capturing these different rates of emotional changes. It is important to develop computational models that are sensitive to these varying speeds. While current computational models can analyze data sequences to comprehend evolving emotions, additional research is required to enhance their efficacy in monitoring and predicting emotional variations over time [ 146 ] .

 

 ■ \blacksquare 
 
 Interpretability of Emotion Understanding Models :
The complexity of ML and DL computational models presents challenges in interpreting their internal workings, particularly when analyzing sensitive mental health data [ 62 ] . Future research is required to improve the interpretability of these models to ensure their trustworthiness and effective utilization in mental health contexts [ 128 ] .

 

 ■ \blacksquare 
 
 Individual Cultural Differences :
The variations in styles used by individuals and cultures to express emotions present challenges in their analysis [ 83 ] . For example, diverse skin tones complicate facial analysis for emotion understanding [ 249 ] , while variations in voice annotations affect audio-based emotion analysis [ 89 ] . Addressing this challenge requires acquiring data from diverse populations and developing computational systems that can adapt to individual changes over time [ 146 ] .

 

 ■ \blacksquare 
 
 Context Dependency :
The emotions are not static and can vary widely over time and across different situations. Understanding stress, depression and engagement requires considering the context in which they occur. Constructing models capable of recognizing context is important and it is also essential to design them so that they can adapt to changing contexts in real-time [ 65 ] .

 

 ■ \blacksquare 
 
 Integration of Computational, Psychological, Medical and Social Analysis :
Developing computational methods that merge insights from medicine, psychology and sociology is crucial in mental health research [ 184 ] . Standardized approaches are vital for universal application and recognition in these fields, fostering a unified understanding of mental health and facilitating research translation into practice [ 15 ] . Synergizing insights from these disciplines is key to improving the accuracy and effectiveness of mental health interventions.

 

 
 
 

### VI.1 Discussion of Challenges and Future Directions 

 
 Future research must not only address critical challenges in accuracy and real-world applicability but also capitalize on the dynamic interplay among stress, depression and engagement to develop effective interventions. Mitigating daily stressors can enhance engagement and reduce depressive symptoms, creating a reinforcing cycle that bolsters cognitive and emotional well-being [ 14 ] . Tailoring interventions to individual engagement profiles may help sustain motivation under heightened stress. Meanwhile, advancements in generative AI, such as LLMs and synthetic data augmentation, show promise but remain underexplored here [ 132 ] . Data scarcity demands collaborative efforts to improve dataset quality [ 49 ] . Robust labeling and fusion techniques are needed to handle varied modalities [ 45 ] and model generalization requires domain adaptation for high inter-individual variability [ 99 ] . Additionally, these states evolve over time, necessitating dynamic, context-aware models that adapt to changing contexts [ 11 ] . The black-box nature of ML and DL underscores the need for interpretability and accounting for cultural and contextual factors remains crucial for unified, adaptive mental health analysis [ 62 ] .

 
 
 
 

## VII Conclusions 

 
 This survey is the first to collectively review computational methods for detecting stress, depression and engagement. It traces the evolution from traditional techniques to advanced ML and DL approaches, highlighting their potential to improve mental healthcare with early detection, personalized interventions and ongoing monitoring. We explore the complexities of multimodal datasets and the challenges they introduce, highlighting a shift towards sophisticated algorithms that offer deeper mental health insights. Our review underscores DL’s transition, emphasizing its accuracy despite its computational intensity and substantial data requirements. The incorporation of multimodal data, including wearables and social media analysis, mirrors the innovative direction of current research and the movement towards interpretable methods for application transparency. This paper highlights the transformative role of computational methods in mental healthcare and calls for ongoing innovation to advance personalized and effective interventions.

 
 
 

## Acknowledgment

 
 The authors express gratitude to the Center for Machine Vision and Signal Analysis, University of Oulu, Finland for the academic and literary resources provided under the Research Council of Finland Profi 5 HiDyn grant 24630111132. The work was partially supported by the Eudaimonia Institute of the University of Oulu.

 
 
 

## References

 
 
 [1] 
 
H. R. Kim, Y. S. Kim, et al. , “Building Emotional Machines:
Recognizing Image Emotions Through Deep Neural Networks,” IEEE Transactions on Multimedia , vol. 20, no. 11, pp. 2980–2992, 2018.

 

 
 [2] 
 
M. Kächele, M. Glodek, et al. , “Fusion of Audio-Visual
Features Using Hierarchical Classifier Systems for Recognition of
Affective States and the State of Depression,” depression ,
vol. 1, no. 1, pp. 671–678, 2014.

 

 
 [3] 
 
A. F. Arnsten, “Stress Weakens Prefrontal Networks: Molecular Insults to
Higher Cognition,” Nature Neuroscience , vol. 18, no. 10, 2015.

 

 
 [4] 
 
P. Ekman et al. , “An Argument for Basic Emotions,” Cognition Emotion , vol. 6, no. 3-4, pp. 169–200, 1992.

 

 
 [5] 
 
S. PS and G. Mahalakshmi, “Emotion Models: A Review,” Int.
Journal of Control Theory Applications , vol. 10, no. 8, pp. 651–657,
2017.

 

 
 [6] 
 
M. S. Akhtar, D. Ghosal, et al. , “All-in-One: Emotion, Sentiment
and Intensity Prediction Using a Multi-Task Ensemble
Framework,” IEEE Transactions on Affective Computing , vol. 13,
no. 1, pp. 285–297, 2019.

 

 
 [7] 
 
M. Munezero, C. S. Montero, et al. , “Are They Different? Affect,
Feeling, Emotion, Sentiment, and Opinion Detection in Text,”
 IEEE Transactions on Affective Computing , vol. 5, no. 2, pp. 101–111,
2014.

 

 
 [8] 
 
R. Draghi-Lorenz, V. Reddy, et al. , “Rethinking the Development of
‘Non-basic’ Emotions: A Critical Review of Existing Theories,”
 Developmental Review , vol. 21, no. 3, pp. 263–304, 2001.

 

 
 [9] 
 
K. R. Scherer, “The Dynamic Architecture of Emotion: Evidence for the
Component Process Model,” Cognition and Emotion , vol. 23, no. 7,
pp. 1307–1351, 2009.

 

 
 [10] 
 
I. J. Roseman, M. S. Spindel, and P. E. Jose, “Appraisals of
Emotion-Eliciting Events: Testing a Theory of Discrete Emotions.,” Journal of personality and social psychology , vol. 59, no. 5, p. 899, 1990.

 

 
 [11] 
 
P. M. Niedenthal, “Embodying Emotion,” Science , vol. 316, no. 5827,
pp. 1002–1005, 2007.

 

 
 [12] 
 
L. F. Barrett, How Emotions Are Made: The Secret Life of the Brain .

 
 Pan Macmillan, 2017.

 

 
 [13] 
 
C. Wrosch, R. Schulz, et al. , “Health Stresses and Depressive
Symptomatology in the Elderly: The Importance of Health
Engagement Control Strategies.,” Health Psychology , vol. 21,
no. 4, p. 340, 2002.

 

 
 [14] 
 
D. A. Pizzagalli, “Depression, Stress, and Anhedonia: Toward a Synthesis and
Integrated Model,” Annual Review of Clinical Psychology , vol. 10,
no. 1, pp. 393–423, 2014.

 

 
 [15] 
 
N. Z. Gurel, M. T. Wittbrodt, et al. , “Automatic Detection of
Target Engagement in Transcutaneous Cervical Vagal Nerve
Stimulation for Traumatic Stress Triggers,” IEEE Journal of
Biomedical and Health Informatics , vol. 24, no. 7, pp. 1917–1925, 2020.

 

 
 [16] 
 
V. Maydych et al. , “The Interplay Between Stress,
Inflammation, and Emotional Attention: Relevance for Depression,”
 Frontiers in neuroscience , vol. 13, p. 384, 2019.

 

 
 [17] 
 
G. M. Slavich and M. R. Irwin, “From Stress To Inflammation And Major
Depressive Disorder: A Social Signal Transduction Theory Of Depression,”
 Psychological bulletin , vol. 140, no. 3, p. 774, 2014.

 

 
 [18] 
 
S. T. Innstrand, E. M. Langballe, et al. , “A Longitudinal Study of
the Relationship Between Work Engagement and Symptoms of Anxiety
and Depression,” Stress and health , vol. 28, no. 1, pp. 1–10, 2012.

 

 
 [19] 
 
S. J. Lupien, B. S. McEwen, M. R. Gunnar, and C. Heim, “Effects of Stress
Throughout Lifespan on The Brain, Behaviour And Cognition,” Nature
Reviews Neuroscience , vol. 10, no. 6, pp. 434–445, 2009.

 

 
 [20] 
 
V. Krishnan and E. J. Nestler, “The Molecular Neurobiology of Depression,”
 Nature , vol. 455, no. 7215, pp. 894–902, 2008.

 

 
 [21] 
 
C. Liston, B. S. McEwen, and B. Casey, “Psychosocial Stress Reversibly
Disrupts Prefrontal Processing Attentional Control,” Proceedings of
the National Academy of Sciences , vol. 106, no. 3, pp. 912–917, 2009.

 

 
 [22] 
 
A. S. Heller, T. Johnstone, A. J. Shackman, et al. , “Reduced Capacity
to Sustain Positive Emotion in Major Depression Reflects Diminished
Maintenance of Fronto-Striatal Brain Activation,” National Academy of
Sciences , vol. 106, no. 52, pp. 22445–22450, 2009.

 

 
 [23] 
 
G. Giannakakis, D. Grigoriadis, et al. , “Review on Psychological
Stress Detection Using Biosignals,” IEEE Transactions on
Affective Computing , vol. 13, no. 1, pp. 440–460, 2019.

 

 
 [24] 
 
A. Němcová, V. Svozilová, et al. , “Multimodal Features
for Detection of Driver Stress and Fatigue,” IEEE Transactions
on Intelligent Transportation Systems , vol. 22, no. 6, pp. 3214–3233, 2020.

 

 
 [25] 
 
K. Magtibay and K. Umapathy, “A Review of Tools and Methods For
Detection, Analysis, and Prediction of Allostatic Load Due to
Workplace Stress,” IEEE Transactions on Affective Computing , 2023.

 

 
 [26] 
 
H. Peng, C. Xia, et al. , “Multivariate Pattern Analysis of
EEG-Based Functional Connectivity: A Study on the
Identification of Depression,” IEEE Access , vol. 7,
pp. 92630–92641, 2019.

 

 
 [27] 
 
L. He et al. , “Deep Learning for Depression Recognition with
Audiovisual Cues: A Review,” Information Fusion , vol. 80,
pp. 56–86, 2022.

 

 
 [28] 
 
J. Lipschitz et al. , “Adoption of Mobile Apps for Depression and
Anxiety: Cross-Sectional Survey Study on Patient Interest and
Barriers to Engagement,” JMIR mental health , vol. 6, no. 1,
p. e11334, 2019.

 

 
 [29] 
 
M. Perkmann et al. , “Academic Engagement: A Review of the
Literature 2011-2019,” Research policy , vol. 50, no. 1, p. 104114,
2021.

 

 
 [30] 
 
A. M. Saks, J. A. Gruman, et al. , “Organization Engagement: A
Review and Comparison to Job Engagement,” Journal of
Organizational Effectiveness: People and Performance , vol. 9, no. 1,
pp. 20–49, 2022.

 

 
 [31] 
 
H. Salam et al. , “Automatic Context-Driven Inference of
Engagement in HMI: A Survey,” arXiv preprint arXiv:2209.15370 ,
2022.

 

 
 [32] 
 
M. MG et al. , “The Role of Engagement in
Teleneurorehabilitation: A Systematic Review,” Frontiers in
Neurology , vol. 11, p. 354, 2020.

 

 
 [33] 
 
A. Abedi, S. Khan, et al. , “Affect-Driven Ordinal Engagement
Measurement from Video,” arXiv preprint arXiv:2106.10882 , 2021.

 

 
 [34] 
 
W. C. De Melo et al. , “Encoding Temporal Information for
Automatic Depression Recognition from Facial Analysis,” in IEEE Int. Conf. on Acoustics, Speech and Signal Processing , pp. 1080–1084,
2020.

 

 
 [35] 
 
Q. Chen, I. Chaturvedi, et al. , “Sequential Fusion of Facial
Appearance and Dynamics for Depression Recognition,” Pattern
Recognition Letters , vol. 150, pp. 115–121, 2021.

 

 
 [36] 
 
J. Li et al. , “Intelligent Depression Detection with
Asynchronous Federated Optimization,” Complex Intelligent
Sys. , pp. 1–17, 2022.

 

 
 [37] 
 
X. Xu, H. Peng, et al. , “Privacy-Preserving Federated Depression
Detection from Multisource Mobile Health Data,” IEEE
Transactions on Industrial Informatics , vol. 18, no. 7, pp. 4788–4797,
2021.

 

 
 [38] 
 
M. Niu et al. , “HCAG: A Hierarchical Context-Aware
Graph Attention Model For Depression Detection,” in IEEE
International Conference on Acoustics, Speech and Signal Processing ,
pp. 4235–4239, 2021.

 

 
 [39] 
 
S. Suparatpinyo and N. Soonthornphisaj, “Smart Voice Recognition Based
on Deep Learning for Depression Diagnosis,” Artificial Life and
Robotics , pp. 1–11, 2023.

 

 
 [40] 
 
S. Gupta, P. Kumar, et al. , “Facial Emotion Recognition Based
Real-Time Learner Engagement Detection System in Online
Learning Context Using Deep Learning Models,” Multimedia
Tools and Applications , vol. 82, no. 8, pp. 11365–11394, 2023.

 

 
 [41] 
 
S. Gupta, P. Kumar, et al. , “A Multimodal Facial Cues Based
Engagement Detection System in E-learning Context Using Deep
Learning Approach,” Multimedia Tools and Applications , pp. 1–27,
2023.

 

 
 [42] 
 
Z. Kastrati et al. , “Soaring Energy Prices: Understanding
Public Engagement on Twitter Using Sentiment Analysis and Topic
Modeling With Transformers,” IEEE Access , vol. 11,
pp. 26541–26553, 2023.

 

 
 [43] 
 
X. Tao, A. Shannon-Honson, et al. , “Towards Understanding the
Engagement Emotional Behaviour of MOOC Students Using
Sentiment Semantic Features,” Computers and Education ,
p. 100116, 2023.

 

 
 [44] 
 
A. V. Savchenko et al. , “Classifying Emotions and Engagement in
Online Learning Based on Single FER Neural Network,” IEEE
Transactions on Affective Computing , vol. 13, no. 4, pp. 2132–2143, 2022.

 

 
 [45] 
 
N. K. Mehta, S. S. Prasad, et al. , “Three-Dimensional DenseNet
Self-Attention Neural Net for Automatic Detection of
Student’s Engagement,” Applied Intelligence , vol. 52, no. 12,
pp. 13803–13823, 2022.

 

 
 [46] 
 
T. Selim, I. Elkabani, et al. , “Students Engagement Level
Detection in Online E-learning Using Hybrid EfficientNetB7
Together with TCN, LSTM, and BI-LSTM,” IEEE Access , vol. 10,
pp. 99573–99583, 2022.

 

 
 [47] 
 
O. Copur, M. Nakıp, et al. , “Engagement Detection with
Multi-task Training in E-learning Environments,” in Image
Analysis and Processing , pp. 411–422, Springer, 2022.

 

 
 [48] 
 
S. Khenkar, S. K. Jarraya, et al. , “Engagement Detection Based on
Analyzing Micro Body Gestures Using 3D CNN.,” Computers, Materials Continua , vol. 70, no. 2, 2022.

 

 
 [49] 
 
T. Lee, D. Kim, et al. , “Predicting Mind-Wandering with Facial
Videos in Online Lectures,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 2104–2113, 2022.

 

 
 [50] 
 
H. Chen et al. , “SMG: A Micro-Gesture Dataset Towards
Spontaneous Body Gestures for Emotional Stress State
Analysis,” International Journal of Computer Vision , vol. 131,
no. 6, pp. 1346–1366, 2023.

 

 
 [51] 
 
S. Carrizosa-Botero et al. , “A Systematic Review of Thermal and
Cognitive Stress Indicators: Implications for Use Scenarios on
Sensor-Based Stress Detection,” in INTERACT , pp. 73–92,
Springer, 2021.

 

 
 [52] 
 
P. Qi, D. Chiaro, et al. , “A Blockchain-based Secure Internet of
Medical Things Framework for Stress Detection,” Information
Sciences , vol. 628, pp. 377–390, 2023.

 

 
 [53] 
 
F. Delmastro, F. Di Martino, et al. , “Cognitive Training and
Stress Detection in Mci Frail Older People Through Wearable
Sensors and Machine Learning,” IEEE Access , vol. 8,
pp. 65573–65590, 2020.

 

 
 [54] 
 
Z. Sun, A. Vedernikov, et al. , “Estimating Stress in Online
Meetings by Remote Physiological Signal and Behavioral
Features,” in ACM UbiComp / ISWC 2022 , pp. 216–220.

 

 
 [55] 
 
M. N. Rastgoo, B. Nakisa, et al. , “A Critical Review of
Proactive Detection of Driver Stress Levels Based on Multimodal
Measurements,” ACM Computing Surveys , vol. 51, no. 5, pp. 1–35,
2018.

 

 
 [56] 
 
Y. S. Can, B. Arnrich, et al. , “Stress Detection in Daily Life
Scenarios Using Smart Phones and Wearable Sensors: A
Survey,” Journal of biomedical informatics , vol. 92, p. 103139,
2019.

 

 
 [57] 
 
T. A. Roldán-Rojo, E. Rendón-Veléz, et al. , “Stressors and
Algorithms Used for Stress Detection: A Review,” in 9th
Int. Conf. on Affective Computing and Intelligent Interaction , pp. 1–8,
IEEE, 2021.

 

 
 [58] 
 
P. Theerthagiri et al. , “Stress Emotion Recognition with
Discrepancy Reduction Using Transfer Learning,” Multimedia
Tools and Applications , vol. 82, no. 4, pp. 5949–5963, 2023.

 

 
 [59] 
 
E. Turcan, K. McKeown, et al. , “Dreaddit: A Reddit Dataset for
Stress Analysis in Social Media,” EMNLP-IJCNLP 2019 , p. 97,
2019.

 

 
 [60] 
 
P. Schmidt, A. Reiss, et al. , “Introducing WESAD, a
Multimodal Dataset for Wearable Stress and Affect Detection,” in
 ACM International Conference on Multimodal Interaction , pp. 400–408,
2018.

 

 
 [61] 
 
J. S. Banerjee, M. Mahmud, et al. , “Heart Rate Variability-Based
Mental Stress Detection: An Explainable Machine Learning
Approach,” SN Computer Science , vol. 4, no. 2, p. 176, 2023.

 

 
 [62] 
 
M. Naegelin et al. , “An Interpretable Machine Learning
Approach to Multimodal Stress Detection in a Simulated Office
Environment,” Journal of Biomedical Informatics , vol. 139,
p. 104299, 2023.

 

 
 [63] 
 
L. He, D. Jiang, et al. , “Automatic Depression Analysis Using
Dynamic Facial Appearance Descriptor and Dirichlet Process
Fisher Encoding,” IEEE Transactions on Multimedia , vol. 21, no. 6,
pp. 1476–1486, 2018.

 

 
 [64] 
 
X. Li et al. , “A Spontaneous Micro Expression Database:
Inducement, Collection and Baseline,” in IEEE International
Conference on Automatic Face and Gesture Recognition , pp. 1–6, 2013.

 

 
 [65] 
 
J. Liao, Y. Liang, et al. , “Deep Facial Spatiotemporal Network
for Engagement Prediction in Online Learning,” Applied
Intelligence , vol. 51, pp. 6609–6621, 2021.

 

 
 [66] 
 
R. B. R. Maddu and S. Murugappan, “Online Learners’ Engagement Detection
via FER in Online Learning Context using Hybrid Model,” Social Network
Analysis and Mining , vol. 14, no. 1, p. 43, 2024.

 

 
 [67] 
 
C. Viegas et al. , “Towards Independent Stress Detection: A
Dependent Model Using Facial Action Units,” in International Conference on Content-based Multimedia Indexing , pp. 1–6,
IEEE, 2018.

 

 
 [68] 
 
I. Alkabbany, A. Ali, A. Farag, I. Bennett, M. Ghanoum, and A. Farag,
“Measuring Student Engagement Level Using Facial Information,”
in International Conf. on Image Processing , pp. 3337–3341, IEEE, 2019.

 

 
 [69] 
 
H. Akbar et al. , “Exploiting Facial Action Unit in Video for
Recognizing Depression using Metaheuristic and Neural Networks,”
in Int. Conf. on Computer Science and AI , vol. 1, pp. 438–443, IEEE,
2021.

 

 
 [70] 
 
J. Shen, H. Yang, et al. , “Assessing Learning Engagement Based
on Facial Expression Recognition in MOOC’s Scenario,” Multimedia Systems , pp. 1–10, 2022.

 

 
 [71] 
 
Y. Shan et al. , “Respiratory Signal and Human Stress:
Non-Contact Detection of Stress with a Low-Cost Depth Sensing
Camera,” International Journal of ML and Cybernetics , vol. 11,
no. 8, pp. 1825–1837, 2020.

 

 
 [72] 
 
Y. Choi, J. Kim, et al. , “Immersion Measurement in Watching
Videos Using Eye-tracking Data,” IEEE Transactions on Affective
Computing , vol. 13, no. 4, pp. 1759–1770, 2022.

 

 
 [73] 
 
R. Kuttala, R. Subramanian, et al. , “Multimodal hierarchical CNN
feature fusion for stress detection,” IEEE Access , 2023.

 

 
 [74] 
 
S. Alghowinem et al. , “Multimodal Depression Detection: Fusion
Analysis of Paralinguistic, Head Pose and Eye Gaze Behaviors,”
 IEEE Transactions on Affective Computing , vol. 9, no. 4, pp. 478–490,
2016.

 

 
 [75] 
 
T. Ashwin and R. M. R. Guddeti, “Affective Database for e-Learning and
Classroom Environments using Indian Students’ Faces, Hand
Gestures and Body Postures,” Future Generation Computer Systems ,
vol. 108, pp. 334–348, 2020.

 

 
 [76] 
 
C. Chang, C. Zhang, L. Chen, Y. Liu, et al. , “An Ensemble Model
Using Face and Body Tracking for Engagement Detection,” in 20th ACM Int. Conf. on Multimodal Interaction , p. 616–622, ACM,
2018.

 

 
 [77] 
 
H. Chen et al. , “Analyze Spontaneous Gestures for Emotional
Stress State Recognition: A Micro-Gesture Data and
Analysis,” in 14th IEEE Int. Conf. on Automatic Face Gesture
Recognition , pp. 1–8, 2019.

 

 
 [78] 
 
J. R. M. Fernández, L. Anishchenko, et al. , “Mental Stress
Detection Using Bioradar Respiratory Signals,” Biomedical
Signal Processing and Control , vol. 43, pp. 244–249, 2018.

 

 
 [79] 
 
L. Zhu et al. , “Stress Detection Through Wrist-Based
Electrodermal Activity Monitoring and Machine Learning,” Journal of Biomedical and Health Informatics , 2023.

 

 
 [80] 
 
L. Xia, A. S. Malik, et al. , “A Physiological Signal-based
Method for Early Mental-Stress Detection,” Biomedical Signal
Processing and Control , vol. 46, pp. 18–32, 2018.

 

 
 [81] 
 
E. A. Sağbaş, S. Korukoglu, et al. , “Stress Detection
via Keyboard Typing Behaviors by Using Smartphone Sensors and
Machine Learning Techniques,” Journal of Medical Systems ,
vol. 44, pp. 1–12, 2020.

 

 
 [82] 
 
P. Zontone et al. , “Stress Detection Through Electrodermal
Activity and Electrocardiogram Analysis in Car Drivers,” in 27th European Signal Processing Conference , pp. 1–5, IEEE, 2019.

 

 
 [83] 
 
T. M. Ober, C. J. Brenner, et al. , “Detecting Patterns of
Engagement in a Digital Cognitive Skills Training Game,” Computers Education , vol. 165, p. 104144, 2021.

 

 
 [84] 
 
Y. Yu, S. Ding, et al. , “Cloud-Edge Collaborative Depression
Detection Using Negative Emotion Recognition and Cross-Scale
Facial Feature Analysis,” IEEE Transactions on Industrial
Informatics , 2022.

 

 
 [85] 
 
M. Migovich et al. , “Stress Detection of Autistic Adults during
Simulated Job Interviews Using a Novel Physiological Dataset and ML,” ACM Transactions on Accessible Computing , vol. 17, no. 1, 2024.

 

 
 [86] 
 
M. Sawadogo et al. , “PTSD in the Wild: A Video Database for Studying
Post-Traumatic Stress Disorder in Unconstrained Environments,” Multimedia Tools and Applications , vol. 83, no. 14, 2024.

 

 
 [87] 
 
M. Kerasiotis, L. Ilias, and D. Askounis, “Depression Detection in Social
Media Posts using Transformer Models and Auxiliary Features,” Social
Network Analysis and Mining , vol. 14, no. 1, p. 196, 2024.

 

 
 [88] 
 
Z. Huang, J. Epps, et al. , “Domain Adaptation for Enhancing
Speech-Based Depression Detection in Natural Environmental
Conditions Using Dilated CNNs.,” in INTERSPEECH ,
pp. 4561–4565, 2020.

 

 
 [89] 
 
S. Sardari, B. Nakisa, et al. , “Audio Based Depression Detection
Using Convolutional Autoencoder,” Expert Systems with
Applications , vol. 189, p. 116076, 2022.

 

 
 [90] 
 
A. Othmani, D. Kadoch, et al. , “Towards Robust Deep Neural
Networks for Affect and Depression Recognition from Speech,” in
 26th International Conference on Pattern Recognition , pp. 5–19,
Springer, 2021.

 

 
 [91] 
 
Z. Huang, J. Epps, et al. , “Depression Detection from Short
Utterances via Diverse Smartphones in Natural Environmental
Conditions,” in INTERSPEECH , pp. 3393–3397, 2018.

 

 
 [92] 
 
F. M. Talaat, “Explainable Enhanced Recurrent Neural Network for Lie
Detection Using Voice Stress Analysis,” Multimedia Tools and
Applications , vol. 83, no. 11, pp. 32277–32299, 2024.

 

 
 [93] 
 
G. Dogan and F. P. Akbulut, “Multimodal Fusion Through Biosignal, Audio and
Visual Content for Detection of Mental Stress,” Neural Computing and
Applications , vol. 35, no. 34, pp. 24435–24454, 2023.

 

 
 [94] 
 
R. Chiong et al. , “A Textual-Based Featuring Approach for
Depression Detection Using Machine Learning Classifiers and
Social Media Texts,” Computers in Biology and Medicine ,
vol. 135, p. 104499, 2021.

 

 
 [95] 
 
M. De Choudhury, M. Gamon, et al. , “Predicting Depression Via
Social Media,” in International AAAI Conference on Web and Social
Media , vol. 7, pp. 128–137, 2013.

 

 
 [96] 
 
H. Zogan et al. , “Hierarchical Convolutional Attention Network
for Depression Detection on Social Media and Its Impact During
Pandemic,” Journal of Biomedical and Health Informatics , 2023.

 

 
 [97] 
 
W. Ragheb, J. Azé, et al. , “Negatively Correlated Noisy
Learners for At-Risk User Detection on Social Networks: A
Study on Depression, Anorexia, Self-Harm, and Suicide,” IEEE Transactions on Knowledge and Data Engineering , vol. 35, no. 1,
pp. 770–783, 2021.

 

 
 [98] 
 
E. de León and D. Trilling, “A Sadness Bias in Political News
Sharing? The Role of Discrete Emotions in Engagement 
Dissemination of Political Facebook News,” Social Media +
Society , vol. 7, no. 4, 2021.

 

 
 [99] 
 
S. Benini, M. Savardi, K. Balint, et al. , “On the Influence of
Shot Scale on Film Mood and Narrative Engagement in Film
Viewers,” IEEE Transactions on Affective Computing , vol. 13, no. 2,
pp. 592–603, 2019.

 

 
 [100] 
 
S. A. Khowaja et al. , “Depression Detection From Social Media Posts
Using Emotion Aware Encoders and Fuzzy Based Contrastive Networks,” IEEE Transactions on Fuzzy Systems , 2024.

 

 
 [101] 
 
J. C. Liu et al. , “Learning from Others without Sacrificing
Privacy: Simulation Comparing Centralized and Federated Ml on
Mobile Health Data,” JMIR mHealth and uHealth , vol. 9, no. 3,
p. e23728, 2021.

 

 
 [102] 
 
P. Bobade and M. Vani, “Stress Detection with Machine and Deep
Learning Using Multimodal Physiological Data,” in 2nd Int.
Conf. on Inventive Research in Computing Applications , pp. 51–57, IEEE,
2020.

 

 
 [103] 
 
B. Zhu, X. Lan, X. Guo, K. E. Barner, C. Boncelet, et al. , “Multi-rate
Attention Based GRU Model for Engagement Prediction,” in International Conference on Multimodal Interaction , pp. 841–848, 2020.

 

 
 [104] 
 
O. AlZoubi, S. K. D’Mello, et al. , “Detecting Naturalistic
Expressions of Nonbasic Affect Using Physiological Signals,”
 IEEE Transactions on Affective Computing , vol. 3, no. 3, pp. 298–310,
2012.

 

 
 [105] 
 
L. Stappen et al. , “The MuSe 2021 Multimodal Sentiment
Analysis Challenge: Sentiment, Emotion, Physiological-Emotion,
and Stress,” in 2nd on Multimodal Sentiment Analysis Challenge ,
pp. 5–14, 2021.

 

 
 [106] 
 
L. Christ et al. , “The Muse 2022 Multimodal Sentiment Analysis
Challenge: Humor, Emotional Reactions, and Stress,” in Int.
on Multimodal Sentiment Analysis Workshop and Challenge , pp. 5–14, 2022.

 

 
 [107] 
 
M. Niu, J. Tao, et al. , “Multimodal Spatiotemporal Representation
for Automatic Depression Level Detection,” IEEE Transactions on
Affective Computing , 2020.

 

 
 [108] 
 
M. Fang, S. Peng, et al. , “A Multimodal Fusion Model with
Multilevel Attention Mechanism for Depression Detection,” Biomedical Signal Processing and Control , vol. 82, p. 104561, 2023.

 

 
 [109] 
 
C. Á. Casado, M. L. Cañellas, et al. , “Depression
Recognition Using Remote Photoplethysmography from Facial
Videos,” IEEE Transactions on Affective Computing , 2023.

 

 
 [110] 
 
M. Rodrigues Makiuchi et al. , “Multimodal Fusion of BERT-CNN and
Gated CNN Representations for Depression Detection,” in 9th
Int. on Audio/Visual Emotion Challenge and Workshop , pp. 55–63, 2019.

 

 
 [111] 
 
Z. Li, Z. An, et al. , “MHA: a Multimodal Hierarchical Attention
Model for Depression Detection in Social Media,” Health
Information Science and Systems , vol. 11, no. 1, p. 6, 2023.

 

 
 [112] 
 
K. Zheng et al. , “Two Birds With One Stone:
Knowledge-Embedded Temporal Graph Net Using Clinical Notes
for Patient Clinical Diagnosis,” Journal of Biomedical
Informatics , vol. 125, pp. 104–116, 2023.

 

 
 [113] 
 
X. Zhang, J. Shen, et al. , “Multimodal Depression Detection:
Fusion of Electroencephalography and Paralinguistic Behaviors Using
a Novel Strategy for Classifier Ensemble,” IEEE Journal of
Biomedical and Health Informatics , vol. 23, no. 6, pp. 2265–2275, 2019.

 

 
 [114] 
 
C. Lin et al. , “SenseMood: Depression Detection on Social
Media,” in International Conference on Multimedia Retrieval ,
pp. 407–411, 2020.

 

 
 [115] 
 
O. Celiktutan et al. , “Multimodal Human-Human-Robot Interactions
(MHHRI) Dataset For Studying Personality And
Engagement,” IEEE Transactions on Affective Computing , vol. 10,
no. 4, pp. 484–497, 2017.

 

 
 [116] 
 
I. Arapakis et al. , “Interest as a Proxy of Engagement in News
Reading: Spectral Entropy Analyses of EEG Activity
Patterns,” IEEE Transactions on Affective Computing , vol. 10, no. 1,
pp. 100–114, 2017.

 

 
 [117] 
 
A. Psaltis, K. C. Apostolakis, et al. , “Multimodal Student
Engagement Recognition in Prosocial Games,” IEEE Transactions
on Games , vol. 10, no. 3, pp. 292–303, 2017.

 

 
 [118] 
 
A. Ben-Youssef et al. , “UE-HRI: A New Dataset For The
Study of User Engagement in Spontaneous Human-Robot
Interactions,” in 19th ACM International Conf. on Multimodal
Interaction , pp. 464–472, 2017.

 

 
 [119] 
 
J. Chen et al. , “IIFDD: Intra and Inter-Modal Fusion
For Depression Detection With Multi-Modal Information From
Internet of Medical Things,” Information Fusion , vol. 102,
p. 102017, 2024.

 

 
 [120] 
 
Y. Xia et al. , “A Depression Detection Model Based on
Multimodal Graph Neural Net,” Multimedia Tools and
Applications , pp. 1–17, 2024.

 

 
 [121] 
 
N. K. Iyortsuun et al. , “Additive Cross-Modal Attention Network
(ACMA) For Depression Detection Based on Audio and
Textual Features,” IEEE Access , 2024.

 

 
 [122] 
 
Y. Tao, M. Yang, H. Li, Y. Wu, and B. Hu, “DepMSTAT: Multimodal
Spatio-Temporal Attentional Transformer for Depression Detection,” IEEE Transactions on Knowledge and Data Engineering , 2024.

 

 
 [123] 
 
C. Yang et al. , “MultiMediate 2023: Engagement Level Detection
Using Audio and Video Features,” in ACM International
Conference on Multimedia , pp. 9601–9605, 2023.

 

 
 [124] 
 
Ö. Sümer et al. , “Multimodal Engagement Analysis From
Facial Videos In The Classroom,” IEEE Transactions on
Affective Computing , 2023.

 

 
 [125] 
 
M. Li, Y. Wei, Y. Zhu, S. Wei, and B. Wu, “Enhancing Multimodal Depression
Detection With Intra-and Inter-Sample Contrastive Learning,” Information Sciences , vol. 684, p. 121282, 2024.

 

 
 [126] 
 
M. Awada, B. Becerik-Gerber, G. Lucas, S. Roll, and R. Liu, “A New
Perspective on Stress Detection: An Automated Approach for Detecting Eustress
 Distress,” IEEE Transactions on Affective Computing , 2023.

 

 
 [127] 
 
S. Mukhopadhyay et al. , “TinyStressNAS: Automated Feature Selection and
Model Generation for On-device Stress Detection,” in ACM Int. Joint
Conf. on Pervasive Ubiquitous Computing , pp. 430–436, 2024.

 

 
 [128] 
 
X. Zhou, K. Jin, et al. , “Visually Interpretable Representation
Learning for Depression Recognition From Facial Images,” IEEE Transactions on Affective Computing , vol. 11, no. 3, pp. 542–552,
2018.

 

 
 [129] 
 
W. C. De Melo, E. Granger, et al. , “Depression Detection Based on
Deep Distribution Learning,” in IEEE International Conference on
Image Processing , pp. 4544–4548, 2019.

 

 
 [130] 
 
R. Tanwar et al. , “A Hybrid Transposed Attention based Deep Learning
Model for Wearable and Explainable Stress Recognition,” Computers and
Electrical Engineering , vol. 119, p. 109551, 2024.

 

 
 [131] 
 
M. Ciman, K. Wac, et al. , “Individuals’ Stress Assessment
Using Human-Smartphone Interaction Analysis,” IEEE
Transactions on Affective Computing , vol. 9, no. 1, pp. 51–65, 2016.

 

 
 [132] 
 
L. Lange, N. Wenzlitschke, and E. Rahm, “Generating Synthetic Health Sensor
Data for Privacy-Preserving Wearable Stress Detection,” Sensors ,
vol. 24, no. 10, p. 3052, 2024.

 

 
 [133] 
 
L. He, J. C.-W. Chan, et al. , “Automatic Depression Recognition
Using CNN with Attention Mechanism from Videos,” Neurocomputing , vol. 422, pp. 165–175, 2021.

 

 
 [134] 
 
A. TS, R. M. R. Guddeti, et al. , “Automatic Detection of
Students’ Affective States in Classroom Environment Using
Hybrid CNNs,” Education and Info Technologies , vol. 25, no. 2,
pp. 1387–1415, 2020.

 

 
 [135] 
 
A. Gupta, A. D’Cunha, et al. , “DAiSEE: Towards User
Engagement Recognition in the Wild,” arXiv preprint
arXiv:1609.01885 , 2016.

 

 
 [136] 
 
M. Quadrini et al. , “Stress Detection with Encoding Physiological
Signals and CNN,” Machine Learning , pp. 1–29, 2024.

 

 
 [137] 
 
D. Giakoumis et al. , “Automatic Recognition Of Boredom In
Video Games Using Novel Biosignal Moment-Based Features,”
 IEEE Transactions on Affective Computing , vol. 2, no. 3, pp. 119–133,
2011.

 

 
 [138] 
 
V. Adarsh and G. Gangadharan, “Mental Stress Detection From
Ultra-Short HRV Using Explainable Graph Convolutional Network
With Network Pruning and Quantisation,” Machine Learning ,
pp. 1–28, 2024.

 

 
 [139] 
 
J. Xu, H. Gunes, et al. , “Two-Stage Temporal Modelling Framework for
Video-based Depression Recognition using Graph Representation,” IEEE
Transactions on Affective Computing , 2024.

 

 
 [140] 
 
L. Yang et al. , “Automatic Feature Learning Combining
Functional Connectivity Net and Graph Regularization for
Depression Detection,” Biomedical Signal Processing and Control ,
vol. 82, p. 104520, 2023.

 

 
 [141] 
 
H. Lu, Z. You, Y. Guo, and X. Hu, “MAST-GCN: Multi-Scale Adaptive
Spatial-Temporal GCN for EEG-Based Depression Recognition,” IEEE
Transactions on Affective Computing , 2024.

 

 
 [142] 
 
E. E. Soares et al. , “The Effects of Engagement with an Online
Depression Prevention for Adolescents on Suicide Risk Factors,”
 Journal of Technology in Behavioral Science , vol. 7, no. 3,
pp. 307–314, 2022.

 

 
 [143] 
 
A. Ragolta et al. , “Hierarchical Attention Net-based Depression
Detection from Transcribed Clinical Interviews,” in INTERSPEECH , 2019.

 

 
 [144] 
 
P.-C. Lin, J.-L. Li, et al. , “In-The-Wild Physiological-Based Stress
Detection Using Federated Strategy,” in IEEE International Conference
on Acoustics, Speech and Signal Processing , pp. 1681–1685, 2024.

 

 
 [145] 
 
J. Ohse, B. Hadžić, et al. , “Zero-Shot Strike: Testing the
Generalisation Capabilities of Out-Of-The-Box LLM Models for Depression
Detection,” Computer Speech Language , vol. 88, p. 101663, 2024.

 

 
 [146] 
 
T. Ashwin and R. M. R. Guddeti, “Impact of Inquiry Interventions on
Students in E-Learning and Classroom Environments,” User
Modeling and User-Adapted Interaction , vol. 30, no. 5, pp. 759–801, 2020.

 

 
 [147] 
 
F. Papadopoulos et al. , “Do Relative Positions and Proxemics Affect the
Engagement in a Human-Robot Collaborative Scenario?,” Interaction
Studies , vol. 17, no. 3, pp. 321–347, 2016.

 

 
 [148] 
 
A. Anusha, P. Sukumaran, et al. , “Electrodermal Activity Based
Pre-Surgery Stress Detection Using a Wrist Wearable,” IEEE Journal of Biomedical and Health Informatics , vol. 24, no. 1,
pp. 92–100, 2019.

 

 
 [149] 
 
G. Gordon, S. Spaulding, et al. , “Affective Personalization of a
Social Robot Tutor for Children’s Second Language Skills,”
in Proceedings of the AAAI Conference on Artificial Intelligence ,
vol. 30.

 

 
 [150] 
 
A. J. Magana, S. T. Mubarrat, D. Kao, and B. Benes, “AI-Based Automatic
Detection of Online Teamwork Engagement in Higher Education,” IEEE
Transactions on Learning Technologies , 2024.

 

 
 [151] 
 
H. S. Chiang et al. , “Cognitive Depression Detection
Cyber-Medical System Based on EEG Analysis And DL
Approaches,” Journal of Biomedical and Health Informatics , vol. 27,
no. 2, pp. 608–616, 2022.

 

 
 [152] 
 
S. Kang, W. Choi, et al. , “K-EmoPhone: A Mobile and Wearable
Dataset with In-Situ Emotion, Stress, and Attention Labels,”
 Scientific Data , vol. 10, no. 1, p. 351, 2023.

 

 
 [153] 
 
N. Wiezer et al. , “Serious Gaming Used as Management
Intervention to Prevent Work-related Stress and Raise Engagement
Among Workers,” in Digital Human Modeling and Applications in
Health, Safety, Ergonomics, and Risk Management , pp. 149–158, Springer,
2013.

 

 
 [154] 
 
L. Mou, C. Zhou, et al. , “Driver Stress Detection Via
Multimodal Fusion Using Attention Based CNN-LSTM,” Expert
Systems with Applications , vol. 173, p. 114693, 2021.

 

 
 [155] 
 
T. Galanti, G. Guidetti, et al. , “Work from Home during the
COVID-19 Outbreak: The Impact on Employees’ Remote Work
Productivity, Engagement, and Stress,” Journal of Occupational
and Environmental Medicine , vol. 63, no. 7, p. 426, 2021.

 

 
 [156] 
 
D. J. a. Travis, “I’m So Stressed!: A Longitudinal Model of
Stress, Burnout Engagement Among Social Workers in Child
Welfare,” The British Journal of Social Work , vol. 46, no. 4,
pp. 1076–1095, 2016.

 

 
 [157] 
 
K. Upadyaya et al. , “From Job Demands Resources to Work
Engagement, Burnout, Life-Satisfaction, Depression, 
Occupational Health,” Burnout Research , vol. 3, no. 4,
pp. 101–108, 2016.

 

 
 [158] 
 
E. O’Brien et al. , “Life-satisfaction, Engagement, Mindfulness,
Flourishing, and Social Support: Do They Predict Depression,
Suicide Ideation, and History of Suicide Attempt in Late
Life?,” The American Journal of Geriatric Psychiatry , vol. 31,
no. 6, pp. 415–424, 2023.

 

 
 [159] 
 
A. Jan et al. , “AI System for Automatic Depression Level
Analysis Through Visual Vocal Expressions,” IEEE
Transactions on Cognitive and Developmental Systems , vol. 10, no. 3,
pp. 668–680, 2017.

 

 
 [160] 
 
A. Kaur et al. , “Prediction and Localization of Student
Engagement in the Wild,” in 2018 Digital Image Computing:
Techniques and Applications , pp. 1–8, IEEE, 2018.

 

 
 [161] 
 
Y. Cho, “Automated Mental Stress Recognition Through Mobile
Thermal Imaging,” in 7th International Conference on Affective
Computing and Intelligent Interaction , pp. 596–600, IEEE, 2017.

 

 
 [162] 
 
G. Lam, H. Dongyan, et al. , “Context-aware Deep Learning for
Multi-modal Depression Detection,” in International Conference on
Acoustics, Speech and Signal Processing , pp. 3946–3950, IEEE, 2019.

 

 
 [163] 
 
X. Chen, L. Niu, et al. , “FaceEngage: Robust Estimation of
Gameplay Engagement From User-contributed Youtube Videos,” IEEE Transactions on Affective Computing , vol. 13, no. 2, pp. 651–665,
2019.

 

 
 [164] 
 
H. Chaptoukaev et al. , “StressID: A Multimodal Dataset for Stress
Identification,” Advances in Neural Information Processing Systems
(NeurIPS) , vol. 36, pp. 29798–29811, 2023.

 

 
 [165] 
 
W.-K. Beh, Y.-H. Wu, et al. , “MAUS: A Dataset for Mental
Workload Assessment on N-back Task Using Wearable Sensor,”
 arXiv preprint arXiv:2111.02561 , 2021.

 

 
 [166] 
 
L. Stappen, A. Baird, et al. , “The Multimodal Sentiment Analysis
in Car Reviews (MuSe-CAR) Dataset: Collection, Insights and
Improvements,” IEEE Transactions on Affective Computing , 2021.

 

 
 [167] 
 
M. Yadav et al. , “Exploring Individual Differences of Public
Speaking Anxiety in Real-Life and Virtual Presentations,” IEEE Transactions on Affective Computing , vol. 13, no. 3, pp. 1168–1182,
2020.

 

 
 [168] 
 
V. Markova, T. Ganchev, et al. , “CLAS: A Database for
Cognitive Load, Affect and Stress Recognition,” in 2019
International Conference on Biomedical Innovations and Applications ,
pp. 1–4, IEEE, 2019.

 

 
 [169] 
 
A. Baghdadi et al. , “DASPS: A Database for Anxious
States Based on a Psychological Stimulation,” arXiv preprint
arXiv:1901.02942 , 2019.

 

 
 [170] 
 
L. Christ et al. , “Multimodal Prediction of Spontaneous Humour:
A Novel Dataset First Results,” arXiv preprint
arXiv:2209.14272 , 2022.

 

 
 [171] 
 
J. Healey et al. , “SmartCar: Detecting Driver Stress,” in
 15th International Conf. on Pattern Recognition. , vol. 4, pp. 218–221,
IEEE, 2000.

 

 
 [172] 
 
C. Fu and Others, “MPDD: Multimodal People Depressive Disorder Challenge,”
in Proceedings of the 2025 ACM Multimedia Conference , (Dublin,
Ireland), 2025.

 

 
 [173] 
 
B. Zou et al. , “Semi-Structural Interview-based Chinese Multimodal
Depression Corpus Towards Automatic Preliminary Screening of Depressive
Disorders,” IEEE Transactions on Affective Computing , vol. 14, no. 4,
pp. 2823–2838, 2022.

 

 
 [174] 
 
Y. Jiang et al. , “MMDA: A Multimodal Dataset for Depression and Anxiety
Detection,” in International Conference on Pattern Recognition ,
pp. 691–702, 2022.

 

 
 [175] 
 
Y. Shen, H. Yang, and L. Lin, “Automatic Depression Detection: An Emotional
Audio-Textual Corpus and a GRU/BILSTM-Based Model,” in IEEE
International Conference on Acoustics, Speech and Signal Processing ,
pp. 7082–7086, 2022.

 

 
 [176] 
 
J. Yoon et al. , “D-Vlog: Multimodal Vlog Dataset for Depression
Detection,” in AAAI Conference on AI , pp. 12226–12234, 2022.

 

 
 [177] 
 
Cai, Hanshu and others, “A Multi-modal Open Dataset for Mental-Disorder
Analysis,” Scientific Data , vol. 9, no. 178, pp. 1–12, 2022.

 

 
 [178] 
 
K. Y. Huang et al. , “Detecting Unipolar Bipolar Depressive
Disorders from Elicited Speech using Latent Affective Structure
Model,” IEEE Transactions on Affective Computing , vol. 11, no. 3,
pp. 393–404, 2018.

 

 
 [179] 
 
F. Ringeval et al. , “AVEC Workshop and Challenge: State of
Mind, Detecting Depression with AI, and Cross-Cultural Affect
Recognition,” in Int. Audio Visual Emotion Challenge and Workshop ,
pp. 3–12, 2019.

 

 
 [180] 
 
H.-C. Shing, S. Nair, et al. , “Expert, Crowdsourced, and Machine
Assessment of Suicide Risk Via Online Postings,” in 5th
Workshop on Computational Linguistics and Clinical Psychology , pp. 25–36,
2018.

 

 
 [181] 
 
D. E. Losada et al. , “Overview of eRisk: Early Risk
Prediction on the Internet.,” Critical Letters in Economics and
Finance , 2019.

 

 
 [182] 
 
J. Gratch et al. , “The Distress Analysis Interview Corpus of
Human and Computer Interviews.,” in LREC , pp. 3123–3128,
Reykjavik, 2014.

 

 
 [183] 
 
S. Alghowinem, R. Goecke, et al. , “From Joyous to Clinically
Depressed: Mood Detection Using Spontaneous Speech,” in IEEE International Conference on Face Gesture Recognition , pp. 1–6,
2012.

 

 
 [184] 
 
G. Coppersmith, M. Dredze, et al. , “CLPsych 2015 Shared Task:
Depression and PTSD on Twitter,” in Workshop on Computational
Linguistics and Clinical Psychology , pp. 31–39, 2015.

 

 
 [185] 
 
M. Valstar et al. , “AVEC 2014: 3D Dimensional Affect and
Depression Recognition Challenge,” in 4th ACM International
Workshop on Audio/Visual Emotion Challenge , pp. 3–10, 2014.

 

 
 [186] 
 
M. Valstar et al. , “AVEC 2013: The Continuous
Audio/Visual Emotion and Depression Recognition Challenge,” in
 3rd ACM International Workshop on Audio/Visual Emotion Challenge ,
pp. 3–10, 2013.

 

 
 [187] 
 
F. Ringeval et al. , “Introducing the RECOLA Multimodal Corpus of
Remote Collaborative and Affective Interactions,” in IEEE Int.
Conf. on Automatic Face and Gesture Recognition , pp. 1–8, 2013.

 

 
 [188] 
 
Y. Yang et al. , “Detecting Depression Severity from Vocal
Prosody,” IEEE Transactions on Affective Comp. , vol. 4, no. 2,
pp. 142–150, 2012.

 

 
 [189] 
 
M. Singh et al. , “DREAMS: Diverse Reactions of Engagement and Attention
Mind States Dataset,” in International Conference on Pattern
Recognition , pp. 163–179, Springer, 2024.

 

 
 [190] 
 
M. Singh et al. , “Do I Have Your Attention? A Large
Scale Engagement Prediction Dataset,” arXiv preprint
arXiv:2302.00431 , 2023.

 

 
 [191] 
 
K. Kroenke, R. L. Spitzer, and J. B. Williams, “The PHQ-9: Validity of a
Brief Depression Severity Measure,” Journal of General Internal
Medicine , vol. 16, no. 9, pp. 606–613, 2001.

 

 
 [192] 
 
G. Giannakakis et al. , “A Stress Recognition System Using
HRV Parameters and ML Techniques,” in 8th International Conf.
on Affective Computing and Intelligent Interaction , pp. 269–272, IEEE,
2019.

 

 
 [193] 
 
L. D. Sharma, V. K. Bohat, et al. , “Evolutionary Inspired Approach
for Mental Stress Detection Using EEG Signal,” Expert
Systems with Applications , vol. 197, p. 116634, 2022.

 

 
 [194] 
 
S. Pourmohammadi, A. Maleki, et al. , “Stress Detection Using
ECG and EMG Signals: A Comprehensive Study,” Computer Methods and Programs in Biomedicine , vol. 193, p. 105482, 2020.

 

 
 [195] 
 
E. Rejaibi et al. , “MFCC-based Recurrent Neural Network for
Automatic Clinical Depression Recognition and Assessment from
Speech,” Biomedical Signal Processing and Control , vol. 71,
p. 103107, 2022.

 

 
 [196] 
 
A. H. Orabi, P. Buddhitha, et al. , “Deep Learning for Depression
Detection of Twitter Users,” in 5th Workshop on Computational
Linguistics and Clinical Psychology , pp. 88–97, 2018.

 

 
 [197] 
 
P. Kumar, S. Malik, et al. , “Interpretable Multimodal Emotion
Recognition Using Hybrid Fusion of Speech and Image Data,”
 Multimedia Tools and Applications , pp. 1–22, 2023.

 

 
 [198] 
 
W. Sanchez, A. Martinez, et al. , “A Predictive Model for Stress
Recognition in Desk Jobs,” Journal of Ambient Intelligence and
Humanized Computing , vol. 14, no. 1, pp. 17–29, 2023.

 

 
 [199] 
 
A. Vedernikov et al. , “Analyzing Participants’ Engagement during Online
Meetings Using Unsupervised Remote Photoplethysmography with Behavioral
Features,” in CVPR Workshops , pp. 389–399, 2024.

 

 
 [200] 
 
Z. Sun and X. Li, “Contrast-Phys+: Unsupervised and Weakly-Supervised
Video-Based Remote Physiological Measurement via Spatiotemporal
Contrast,” IEEE Transactions on Pattern Analysis and Machine
Intelligence , 2024.

 

 
 [201] 
 
P. Sarkar and A. Etemad, “Self-supervised ECG Representation Learning
for Emotion Recognition,” IEEE Transactions on Affective
Computing , vol. 13, no. 3, pp. 1541–1554, 2020.

 

 
 [202] 
 
J. Zhong, Z. Shan, et al. , “Robust Discriminant Feature
Extraction for Automatic Depression Recognition,” Biomedical
Signal Processing and Control , vol. 82, p. 104505, 2023.

 

 
 [203] 
 
Y. Cai, H. Wang, et al. , “Depression Detection on Online Social
Network with Multivariate Time Series Feature of User
Depressive Symptoms,” Expert Systems with Applications , p. 119538,
2023.

 

 
 [204] 
 
M. Albaladejo-González et al. , “Evaluating Different
Configurations of Machine Learning Models and Their Transfer
Learning Capabilities For Stress Detection Using Heart
Rate,” Journal of Ambient Intelligence and Humanized Computing ,
vol. 14, no. 8, pp. 11011–11021, 2023.

 

 
 [205] 
 
A. Pampouchidou et al. , “Automated Facial Video-Based
Recognition of Depression and Anxiety Symptom Severity:
Cross-corpus Validation,” Machine Vision and Applications ,
vol. 31, no. 4, p. 30, 2020.

 

 
 [206] 
 
S. Deldari, H. Xue, et al. , “COCOA: Cross Modality Contrastive
Learning for Sensor Data,” ACM on Interactive, Mobile, Wearable
and Ubiquitous Technologies , vol. 6, no. 3, pp. 1–28, 2022.

 

 
 [207] 
 
Q. Zhao, L. Yang, and N. Lyu, “A Driver Stress Detection Model Via
Data Augmentation Based on Deep Convolutional Recurrent Neural
Network,” Expert Systems with Applications , vol. 238, p. 122056,
2024.

 

 
 [208] 
 
J. Wu, Z. Zhou, et al. , “Multi Feature and Multi-Instance
Learning with Anti-Overfitting Strategy for Engagement Intensity
Prediction,” in Int. Conference on Multimodal Interaction ,
pp. 582–588, 2019.

 

 
 [209] 
 
A. I. Beltrán-Velasco et al. , “Analysis of Psychophysiological
Stress Response in Higher Education Students Undergoing
Clinical Practice Evaluation,” vol. 43, pp. 1–7, 2019.

 

 
 [210] 
 
Q. Chen and B. G. Lee, “DL Models for Stress Analysis in University
Students: A Sudoku Based Study,” Sensors , vol. 23, no. 13,
2023.

 

 
 [211] 
 
Y. Lin et al. , “A Deep Learning Model for Detecting Depression
in Senior Population,” Frontiers in Psychiatry , vol. 13,
p. 1016676, 2022.

 

 
 [212] 
 
A. Li et al. , “A Multimodal-Driven Fusion Data Augmentation Framework
for Emotion Recognition,” IEEE Transactions on AI , 2025.

 

 
 [213] 
 
Z. Wang, W. Wu, and C. Zeng, “Physioformer: Integrating Multimodal
Physiological Signals and Symbolic Regression for Explainable Affective State
Prediction,” arXiv preprint arXiv:2410.11376 , 2024.

 

 
 [214] 
 
M. Saha et al. , “Pulse-PPG: An Open-Source Field-Trained PPG Foundation
Model for Wearable Applications Across Lab and Field Settings,” arXiv
preprint arXiv:2502.01108 , 2025.

 

 
 [215] 
 
S. R. Kothuri and N. RajaLakshmi, “Siamese Capsule Gorilla Troops
Network-based Multimodal Sentiment Analysis for Car Reviews,” Soft
Computing , vol. 28, no. 13, pp. 7627–7647, 2024.

 

 
 [216] 
 
M. Niu et al. , “Depression scale dictionary decomposition framework for
multimodal automatic depression level prediction,” IEEE Transactions on
Circuits and Systems for Video Technology , 2025.

 

 
 [217] 
 
Y. Wang et al. , “Automatic Depression Recognition with An Ensemble of
Multimodal Spatio-Temporal Routing Features,” IEEE Transactions on
Affective Computing , 2025.

 

 
 [218] 
 
S. Li et al. , “Efficient long speech sequence modelling for time-domain
depression level estimation,” in IEEE International Conference on
Acoustics, Speech and Signal Processing , pp. 4235–4239, 2025.

 

 
 [219] 
 
P. Kumar, S. Misra, et al. , “Multimodal Interpretable Depression
Analysis Using Visual, Physiological, Audio and Textual Data,” in IEEE/CVF Winter Conference on Applications of Computer Vision , 2024.

 

 
 [220] 
 
V. T. Huynh, H. J. Yang, G.-S. Lee, S.-H. Kim, et al. , “Engagement
Intensity Prediction with Facial Behavior Features,” p. 567 –
571, 2019.

 

 
 [221] 
 
S. Mandia et al. , “Transformer-Driven Modeling of Variable Frequency
Features for Classifying Student Engagement in Online Learning,” arXiv
preprint arXiv:2502.10813 , 2025.

 

 
 [222] 
 
W.-L. Zhang et al. , “A Self-Supervised Learning Network for Student
Engagement Recognition from Facial Expressions,” IEEE Transactions on
Circuits and Systems for Video Technology , 2024.

 

 
 [223] 
 
A. Abedi et al. , “Improving state-of-the-art in Detecting Student
Engagement with Resnet and TCN Hybrid Network,” p. 151 – 157,
2021.

 

 
 [224] 
 
A. Alkurdi et al. , “Extending Anxiety Detection from Multimodal
Wearables in Controlled Conditions to Real-World Environments,” Sensors , vol. 25, no. 4, p. 1241, 2025.

 

 
 [225] 
 
Y. Wu, M. Daoudi, and A. Amad, “Transformer-based Self-Supervised Multimodal
Representation Learning for Wearable Emotion Recognition,” IEEE
Transactions on Affective Computing , vol. 15, no. 1, pp. 157–172, 2023.

 

 
 [226] 
 
P. Bota, C. Wang, et al. , “Emotion Assessment Using Feature
Fusion and Decision Fusion Classification Based on Physiological
Data: Are We There Yet?,” Sensors , vol. 20, no. 17, p. 4723,
2020.

 

 
 [227] 
 
D. Jiang, R. Wei, et al. , “A Multitask Learning Framework for
Multimodal Sentiment Analysis,” in International Conference on
Data Mining Workshops , pp. 151–157, IEEE, 2021.

 

 
 [228] 
 
L. Sun, M. Xu, et al. , “Multimodal Emotion Recognition and
Sentiment Analysis via Attention Enhanced Recurrent Model,” in
 Proceedings of the 2nd on Multimodal Sentiment Analysis Challenge ,
pp. 15–20, 2021.

 

 
 [229] 
 
M. Niu et al. , “Multi-Scale Multi-Region Facial
Discriminative Representation For Automatic Depression Level
Prediction,” in IEEE Int. Conf. on Acoustics, Speech Signal
Proc. , pp. 1325–1329, 2021.

 

 
 [230] 
 
M. Al Jazaery, G. Guo, et al. , “Video-based Depression Level
Analysis by Encoding Deep Spatiotemporal Features,” IEEE
Transactions on Affective Computing , vol. 12, no. 1, pp. 262–268, 2018.

 

 
 [231] 
 
Y. Zhu, Y. Shang, et al. , “Automated Depression Diagnosis Based
on Deep Networks to Encode Facial Appearance and Dynamics,” IEEE Transactions on Affective Computing , vol. 9, no. 4, pp. 578–584, 2017.

 

 
 [232] 
 
L. Wen, X. Li, et al. , “Automated Depression Diagnosis Based on
Facial Dynamic Analysis and Sparse Coding,” IEEE Transactions
on Information Forensics and Security , vol. 10, no. 7, pp. 1432–1441, 2015.

 

 
 [233] 
 
H. Kaya and A. Salah, “Eyes Whisper Depression: CCA Based
Multimodal Approach,” in ACM Int. Conf. on Multimedia , pp. 61–64,
2014.

 

 
 [234] 
 
H. Kaya, F. Çilli, et al. , “Ensemble CCA for Continuous
Emotion Prediction,” in 4th International Workshop on Audio/Visual
Emotion Challenge , pp. 19–26, 2014.

 

 
 [235] 
 
M. Singh et al. , “Do I Have Your Attention: A Large Scale Engagement
Prediction Dataset and Baselines,” in 25th International Conference on
Multimodal Interaction , pp. 174–182, 2023.

 

 
 [236] 
 
K. Wang et al. , “Bootstrap Model Ensemble and Rank Loss for
Engagement Intensity Regression,” in ICMI , p. 551–556, 2019.

 

 
 [237] 
 
X. Niu et al. , “Automatic Engagement Prediction with GAP
Feature,” in International Conference on Multimodal Interaction ,
p. 599–603, 2018.

 

 
 [238] 
 
C. Thomas et al. , “Predicting Engagement Intensity in the Wild
Using Temporal Convolutional Network,” in ACM International
Conference on Multimodal Interaction , p. 604–610, ACM, 2018.

 

 
 [239] 
 
A. Singh et al. , “VisioPhysioENet: Multimodal Engagement Detection
using Visual and Physiological Signals,” arXiv:2409.16126 , 2024.

 

 
 [240] 
 
H. Zhang, X. Xiao, T. Huang, et al. , “An Novel End-to-End
Network for Automatic Student Engagement Recognition,” p. 342 –
345, 2019.

 

 
 [241] 
 
A. I. Siam et al. , “Automatic Stress Detection in Car Drivers
Based on Non-Invasive Physiological Signals Using ML
Techniques,” Neural Computing and Applications , vol. 35, no. 17,
pp. 12891–12904, 2023.

 

 
 [242] 
 
K. Hong, G. Liu, et al. , “Classification of the Emotional Stress
and Physical Stress Using Signal Magnification and Canonical
Correlation Analysis,” Pattern Recognition , vol. 77, pp. 140–149,
2018.

 

 
 [243] 
 
A. Vedernikov, P. Kumar, , et al. , “TCCT-Net: Two-Stream Network
Architecture for Fast and Efficient Engagement Estimation via Behavioral
Feature Signals,” in CVPR Workshops , pp. 4723–4732, 2024.

 

 
 [244] 
 
M. Dia, G. Khodabandelou, and A. Othmani, “Paying Attention to Uncertainty: A
Stochastic Multimodal Transformers for Post-Traumatic Stress Disorder
Detection Using Video,” Computer Methods and Programs in Biomedicine ,
vol. 257, p. 108439, 2024.

 

 
 [245] 
 
J. Wang, H. Ouyang, et al. , “The Application of Machine Learning
Techniques in Posttraumatic Stress Disorder: A Systematic Review and
Meta-Analysis,” NPJ Digital Medicine , vol. 7, no. 1, p. 121, 2024.

 

 
 [246] 
 
D. Xezonaki et al. , “Affective Conditioning on Hierarchical
Attention Networks Applied to Depression Detection from
Transcribed Clinical Interviews,” INTERSPEECH , pp. 4556–4560,
2020.

 

 
 [247] 
 
C. Filippini, E. Spadolini, et al. , “Facilitating The
Child–Robot Interaction By Endowing The Robot With The
Capability of Understanding the Child Engagement: The Case of
Mio-Amico Robot,” International Journal of Social Robotics ,
vol. 13, pp. 677–689, 2021.

 

 
 [248] 
 
L. Steinert, F. L. Kölling, et al. , “Evaluation of An
Engagement-Aware Recommender System for People With Dementia,” in ACM
Conference on User Modeling, Adaptation and Personalization , pp. 89–98,
2022.

 

 
 [249] 
 
B. H. Prasetio, H. Tamura, et al. , “The Facial Stress
Recognition Based on Multi-Histogram Features and CNN,” in 2018 IEEE International Conference on Systems, Man, and Cybernetics ,
pp. 881–887, 2018.

 

 
 
 
 
 
 
 | 
 
 
 Puneet Kumar (Member, IEEE) received his B.E. and M.E. degrees in Computer Science in 2014 and 2018, respectively and his Ph.D. from the IIT Roorkee, India in 2022. He has worked at Oracle, Samsung R D and PaiByTwo Pvt. Ltd. and is now a Postdoctoral Researcher at the University of Oulu, Finland. His research interests include Affective Computing, Multimodal and Interpretable AI, Mental Health and Cognitive Neuroscience. He has published in top journals and conferences and received institute medal in M.E., the best thesis award and several best paper awards. For more information, visit his webpage at www.puneetkumar.com . 
 | 

 
 
 
 
 | 
 
 
 Alexander Vedernikov (Member, IEEE) received his M.Sc. from Politecnico di Milano, Italy, in 2018 and his Ph.D. in Mathematics from Skolkovo Institute of Science Technology, Moscow, Russia, in 2022. He is currently with the Center for Machine Vision and Signal Analysis at the University of Oulu, Finland, working in Affective Computing, Facial Analysis and Emotion Understanding. For more details, visit his webpage at www.linkedin.com/in/a-vedernikov . 
 | 

 
 
 
 
 | 
 
 
 Yuwei Chen received B.E. and M.Sc. from Zhejiang University, China (1999, 2002) and Ph.D. in Circuits and Systems from the Chinese Academy of Sciences (2005). He later obtained a Doctor of Tech. in Telecommunication Software from Aalto University, Finland (2020). He contributed to China’s first moon-exploration satellite, Chang’e, developing its echo-detecting laser range sensor and prototyped China’s first airborne pushbroom laser scanner. Currently, he is director general at Advanced Laser Technology Anhui and guest professor at the Chinese Academy of Sciences and Zhejiang University. He works on hyperspectral LiDAR, radar and navigation and has published over 200 papers and 16 patents. For more information, visit his profile at www.researchgate.net/profile/Yuwei-Chen . 
 | 

 
 
 
 
 | 
 
 
 Wenming Zheng (Senior Member, IEEE) received B.S. in Computer Science from Fuzhou University (1997), M.S. from Huaqiao University (2001) and Ph.D. in Signal Processing from Southeast University (2004). Since then, he is with the Research Center for Learning Science, Southeast University. He is a professor at the School of Biological Science and Medical Engineering and the Key Laboratory of Child Development and Learning Science, Ministry of Education. He works on affective computing, pattern recognition, machine learning and computer vision. He is an Associate Editor for IEEE Transactions on Affective Computing and IEEE Transactions on Cognitive and Developmental Systems. For more information, visit his profile at https://ieeexplore.ieee.org/author/37273477300 . 
 | 

 
 
 
 
 | 
 
 
 Xiaobai Li (Senior Member, IEEE) received her B.Sc. and M.Sc. degrees in 2004 and 2007, respectively and her Ph.D. from the University of Oulu, Finland, in 2017. She is a faculty member at the State Key Lab of Blockchain and Data Security, Zhejiang University, China and an Adjunct Professor at the University of Oulu. Her research focuses on Affective Computing, Facial Expression Recognition, Micro-Expression Analysis and Remote Physiological Signal Analysis. She has co-chaired international workshops at CVPR, ICCV, FG and ACM MM and serves as an Associate Editor for IEEE Transactions on Circuits and Systems for Video Technology, Frontiers in Psychology and Image and Vision Computing. For more information, visit her webpage at https://xiaobaili-uhai.github.io/ . 
 |