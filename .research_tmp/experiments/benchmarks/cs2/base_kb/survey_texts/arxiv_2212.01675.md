Facial emotion recognition systems in smart classroom: A survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2212.01675v1 [cs.CV] 03 Dec 2022 
 
 

# Facial emotion recognition systems in smart classroom: A survey

 
 
 Amimi Rajae 1 
 
 E-mail: amimi.rajae@inpt.ac.ma 
 
    
 Radgui Amina 2 
 
 E-mail: radgui@inpt.ac.ma 
 
    
 and
Ibn el haj el hassane 3 
 
 E-mail: ibnelhaj@inpt.ac.ma 
 

 Abstract 
 
 Technology has transformed traditional educational systems
around the globe; integrating digital learning tools into classrooms offers
students better opportunities to learn efficiently and allows the teacher to
transfer knowledge more easily. In recent years, there have been many improvements in smart classrooms. For instance, the integration of facial
emotion recognition systems (FER) has transformed the classroom into an emotionally aware area using the power of machine intelligence and IoT. This paper provides a consolidated survey of the state- of-the-art in the concept of smart classrooms and presents how the application of FER systems significantly takes this concept to the next level.

 
 
 
 Keywords:  smart classrooms, students affect states, FER system, intelligent tutoring, student expressions database
 † † tocauthor: Amimi Rajae, Radgui Amina, Ibn el haj el hassane † † institute: National Institute of Posts and Telecommunications,Rabat, Morocco
 
 and National Institute of Posts and Telecommunications, Rabat, Morocco
 
 and National Institute of Posts and Telecommunications,Rabat, Morocco
 
 
 

## 1 Introduction

 
 The concept of a modern classroom has long ago attracted the interest of many researchers. It dates back to the early 16th century when the Pilgrims Fathers established the first public school in 1635. Since the 1980s, with the development of information technology such as networking, multimedia, and computer science, the classrooms of various schools have become more and more information-based at different levels.

 
 
 Generally, a classroom is defined as an educational space where a teacher transfers knowledge to a group of students; this learning environment is one of the basic elements that influence the quality of education. Therefore, researchers suggest smart classrooms as an innovative approach that gives rise to a new intelligent teaching methodology, which became popular since 2012. The literature presents two visions of this concept. The first approach that has taken good advantage of the joint growth of the computer science and electronics industry concentrates on the feasibility of deploying various intelligent devices in replacement to traditional materials, such as replacing books with optical discs or pen drives, getting rid of shalk board in favor of interactive whiteboard. In his paper ”what is a smart classroom?”, Yi Zhang [ 1 ] notes ” The smart classroom can be classified as a classroom with computers, projectors, multimedia devices (video and DVD), network access, loudspeakers, etc., and capable of adjusting lighting and controlling video streams;”. According to some authors, this concept of a ”technology-rich classroom” has a significant limitation in that it concentrates solely on the design and equipment of the classroom environment, ignoring pedagogy and learning activities [ 2 ] . However, propositions from other studies point to a different insight, [ 3 ] and al. envision making classrooms an emotionally aware environment that emphasizes improving teaching methodologies, this second approach focuses on the pedagogical aspect rather than technology and software design. Kim and al propose integrating machine intelligence and IOT to create a classroom that can listen, see and analyze students’ engagement [ 3 ] .

 
 
 In 2013, Derick Leony and al confirm that ” Many benefits can be obtained if intelligent systems can bring teachers with knowledge about their learner’s emotions” [ 4 ] , emotions may be a fundamental factor that influences learning, as well as a driving force that encourages learning and engagement. As a result,https://www.overleaf.com/project/62ec3930f99c128355b000c7 researchers suggest a variety of approaches for assessing students’ affect states, including textual, visual, vocal, and multimodal methodologies. According to [ 5 ] , the most widely utilized measurement channels are textual and visual. In comparison to the visual channel, textual (based on questionnaires and text analysis) is less innovative, and Facial Emotion Recognition (FER) systems are classified at the top of the visual channel.
In another state of the art, numerous researchers propose performant FER approaches with high accuracy up to 80% [ 6 ] ; this encourages their integration as an efficient method to analyze student affect states.

 
 
 Even though there has been an increasing amount of attention paid to technologies used in smart education, there is no literature that tackles the many elements of employing students’ facial expression recognition systems in smart classrooms. Our article helps understand the concept of future classrooms and their technologies, particularly students’ FER.

 
 
 We organize this paper as follows: In Section 2, we define smart classrooms and we present some of their technologies. In Section 3, we investigate the integration of FER systems to intelligent classrooms; enumerating FER databases and approaches. In Section 4, we elaborate our insight and future works, then in Section 5, we present a brief conclusion.

 
 
 

## 2 Smart classroom: definition and technologies

 

### 2.1 Definition of smart classroom

 
 Smart learning is technology-enhanced learning, it facilitates interaction between students and their instructor and provides learners access to digital resources; it also provides guidance, tips, helpful tools, and recommendations to teach and learn efficiently. This innovative learning system comes with several new ways of learning classified into two categories: 
 Learning via technology: for instance, taking interactive courses via massively open online courses (MOOC), educational games, or intelligent tutoring systems (ITS).
 Learning with a teacher: consists of learning in a technology-enhanced physical classroom, which is generally the concept of Smart classrooms (SC) .

 
 
 There are many definitions of smart classroom; indeed, it is hard to agree on a single definition that is accepted by the scientific community overall.
Authors like [ 1 ] , introduce intelligent classroom as learners-centered environment that supports students, adapts to their learning abilities, and helps teachers transfer their knowledge interactively and easily. Meanwhile, [ 7 ] [ 8 ] and al define a smart classroom as a physical environment containing digital tools, interactive devices, and various technologies to facilitate the activity of teaching and enhance learning. [ 3 ] and al comes up with a definition that goes beyond considering just the possibility of deploying intelligent materials into a physical place, they envision a smart classroom as an emotionally aware environment using real-time sensing and the power of machine intelligence, and in this context, they suggest a new system with advanced technologies and provide directions to deploy it.
 In literature, authors present the concept of the future classroom in their own distinctive way, but the goal remains the same: to enhance education through technology.
A pertinent question to ask here, is, what elements make up a typical intelligent classroom?
In response to this question, [ 8 ] and al. propose a taxonomy of a typical smart classroom, as shown in Figure 1 .

 
 
 Figure 1: Taxonomy of a typical smart classroom 
 
 
 A typical smart classroom provides tools such as desktop computers, digital cameras, recording and casting equipments, interactive whiteboards, etc; [ 9 ] for effective presentation, better assessment, constructive interaction, and comfortable physical environment. In order to present this taxonomy, four components are to consider:

 Smart content and presentation : technology assists the instructor in preparing the content of his courses and presenting it easily and interactively [ 9 ] .

 Smart assessment : includes automated evaluation of students learning capacity; also, it consists of managing students’ attendance and recuperating their feedback to enhance lectures’ quality as presented by [ 10 ] [ 11 ] .

 Smart physical environment : smart classrooms offer a healthy climate by controlling factors like air, temperature, humidity, etc; using sensors and actuators [ 12 ] [ 13 ] [ 14 ] .

 Smart interaction and engagement : it focuses on analyzing the level of students engagement and consists of providing tools to enhance interaction [ 10 ] [ 15 ] [ 16 ] .

 
 
 

### 2.2 Smart classroom technologies

 
 Education technology (edtech) is a multi-billion-dollar business that is rising every year. Many nations in the Organisation for Economic Co-operation and Development (OECD) spend more than 10% of their budgets on education [ 17 ] .
Furthermore, as more nations raise their education investment, public spending on Edtech is expected to rise. Low- and middle-income nations, for example, expect to boost education expenditure from US$ 1.2 trillion to US$3 trillion per year [ 18 ] . According to the Incheon Declaration, nations must devote at least 4% to 6% of their GDP to education, or at least 15% to 20% of public spending to education.
Aside from the predicted expansion in the educational sector, the market for smart classroom technologies is rapidly growing and is strongly linked to advances in computer science, robotics, and machine intelligence.
As indicated in Table 1 , every technology has advantages and disadvantages; the most prevalent constraints of most of these technologies are cost in the first position, followed by technical knowledge concerns ( generally, teachers and students don’t have the required technical knowledge to use those technologies correctly) in the second place. Below are some of the main SC technologies: 
 Interactive Whiteboard (IWB): is an intelligent tool that allows users to manipulate their presentations and project them on a board’s surface using a special pen or simply their hand [ 19 ] . IWB can be used to digitalize operations and tasks or merely as a presentational device [ 20 ] . The use of this device has revolutionized the nature of educational activities; it has the power to reduce the complexity of teaching and offers the instructor more flexibility during presentations. 
 RFID Attendance Management System : RFID stands for Radio Frequency Identification; it is a wireless technology used to track an object then memorize and recuperate data using radio tags [ 21 ] . An RFID attendance system automatically marks students’ presence in the classroom by validating their ID cards on the reader. It is a bright, innovative solution to replace classic attendance registers [ 22 ] , and help teachers gain wasted time verifying students’ presence daily.
 Educational cobot : is a new innovative tool used as a co-worker robot to help with teaching tasks. It contains several sensors, cameras, microphones, and motors so it can listen, see, communicate with students and assist the teacher [ 23 ] . Embedded into classrooms, cobots represent the school of the future and have significantly changed the traditional ways of teaching [ 24 ] .
 Sensors and actuators : consists of installing sensors to collect data from their environment, then sending this data to the cloud to be analyzed, and might decide to act using actuators; for example, a temperature sensor detects and measures hotness and coolness, then sends the information to another device to adjust the temperature. This technology provides an adequate healthy climate by controlling air, temperature, humidity, etc [ 13 ] .
 Augmented Reality (AR): is a system that combines real and virtual worlds, it is a real-world interactive experience in which computer-generated perceptual information enhances real objects [ 25 ] . It was first used in training pilots applications in the 1990s [ 26 ] , and over the years, it has demonstrated a high efficiency when adopted in educational settings; it has been used to enhance many disciplines particularly when students learn subjects with complex, abstract concepts or simply things they find difficult to visualize such like mathematics, geography, anatomy, etc [ 27 ] .
 A Classroom Response System (CRS): is used to collect answers from all students and send them electronically to be analyzed by the teacher, who can graphically display a summary of the gathered data [ 28 ] . CRS is an efficient method to increase classroom interaction [ 29 ] .According to Martyn (2007) [ 30 ] , “One of the best aspects about an CRS is that it encourages students to contribute without fear of being publicly humiliated or of more outspoken students dominating the discussion.”
 Commercial off-the-shelf (COTS) eye tracker : is a gaze-based model that is used to monitor students’ attention and detect their mind wandering in the classroom. It helps the instructor to better understand their interests and evaluate their degree of awareness [ 31 ] [ 32 ] .
 Wearable badges : tracks the wearer’s position, detects when other badge wearers are in range, and can predict emotion from the wearer’s emotional voice tone. It helps the teacher manage the classroom; he can use those badges to detect if a student leaves without permission or when a group of learners make a loud noise that disturbs the rest of their classmates. MIT has developed a wearable badge by Sandy Pentland’s team [ 24 ] [ 17 ] .
 Learning Management System (LMS): is a web-based integrated software; used to create, deliver and track courses and outcomes. It aids educators to develop courses, post announcements, communicate easily and interactively, grade assignments, and assess their students; besides, it allows students to submit their work, participate in discussions, and take quizzes [ 33 ] [ 34 ] [ 35 ] .

 Student facial emotions recognition (FER): is a technology that analyzes expressions using a person’s images; it is part of the affective computing field. Authors uses this technology to predict real-time student feedback during lectures [ 6 ] , which helps the instructor improve the quality of his presentations.

 
 
 Table 1: Merits and limitations of smart classroom technologies 
 
 
 
 
 
 
 
 
 
 
 \raggedcolumns 
 Technology 
 
 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 Merits 
 
 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 Limitations 
 
 
 | 

 
 
 
 
 
 IWB 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 The touchscreen made its use simpler and more effective. 
 
 
 
 \raggedcolumns 
 It is equipped with smart tools such as a pointer, screen capture… 
 
 
 
 \raggedcolumns 
 It provides access to the web. 
 
 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 Expensive: many schools are not able to afford it. 
 
 
 
 \raggedcolumns 
 Teacher training: The school should spend time and money teaching their instructors how to use the equipment correctly. 
 
 
 | 

 
 
 
 RFID 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 Quick and Rapid: it identifies students in seconds. 
 
 
 
 \raggedcolumns 
 Accuracy : it provides more accurate identification. 
 
 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 Expensive : In case of a large strength of students, 
 
 
 
 \raggedcolumns 
 purchasing tags for everyone is costly. 
 
 
 
 \raggedcolumns 
 Not secure : the system is prone to manipulation. 
 
 
 | 

 
 
 
 COBOT 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 Wide Knowledge: it saves a large amount of information. 
 
 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 Technical Disruptions: it can break down at any time. 
 
 
 
 \raggedcolumns 
 Human-machine interaction: it may have trouble in trying to interact with the students. 
 
 
 
 \raggedcolumns 
 Expensive. 
 
 
 | 

 
 
 
 Sensors and actuators 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 Comfort: it provides a healthy and comfortable environment. 
 
 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 Technical support : it needs IT professionals to help set it up and maintain it 
 
 
 | 

 
 
 
 AR 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 It provides outstanding visualizations. 
 
 
 
 \raggedcolumns 
 It Increases Students’ Engagement. 
 
 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 It may presents functionality Issues. 
 
 
 
 \raggedcolumns 
 It is expensive. 
 
 
 | 

 
 
 
 CRS 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 Rapid assessment: It provides outcomes of formative assessment. 
 
 
 
 \raggedcolumns 
 It provides immediate feedback for student. 
 
 
 
 \raggedcolumns 
 Time saving: for instance, Fast grading. 
 
 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 Expensive : it costs an average of $75 / device 
 
 
 
 \raggedcolumns 
 It presents technical problems. 
 
 
 
 \raggedcolumns 
 Ineffective for opinion questions. 
 
 
 | 

 
 
 
 COTS 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 It records real time eye movements and fixations, which can report student reactivity. 
 
 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 It is not able to track all eyes: eye-tracking camera is 
 
 
 
 \raggedcolumns 
 impacted for example by lenses or glasses. 
 
 
 
 \raggedcolumns 
 It costs money, time and labor resources. 
 
 
 
 \raggedcolumns 
 Efficiency: visual attention is not sufficient to interpret 
 
 
 
 \raggedcolumns 
 students’ engagement. 
 
 
 | 

 
 
 
 
 
 
 
 \raggedcolumns 
 Wearable 
 
 
 
 \raggedcolumns 
 badges 
 
 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 It increases the level of security and privacy in classrooms. 
 
 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 Privacy concern: pupils may have some misapprehension 
 
 
 
 \raggedcolumns 
 about their privacy when it comes to wearable devices. 
 
 
 | 

 
 
 
 LMS 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 It saves wasted time on menial tasks like grading papers. 
 
 
 
 \raggedcolumns 
 It gives students access to learning material in one place from any device. 
 
 
 | 
 
 
 
 
 
 
 \raggedcolumns 
 It requires IT and programming knowledge. 
 
 
 | 

 

 
 
 
 
 

## 3 The application of FER system to smart classroom

 

### 3.1 FER approaches used for smart classroom 

 
 Analyzing student’s affective states during a lecture is a pertinent task in smart classroom [ 36 ] . It is crucial to determine student engagement during a lecture in order to measure the effectiveness of teaching pedagogy and enhance the interaction with the instructor. Therefore, in order to get student feedback and achieve his satisfaction, researchers suggest different methods such as: body gesture recognition detected by using Electroencephalography (EEG) signals [ 37 ] , body posture using either cameras or a sensing chair [ 16 ] , hand gestures [ 38 ] , heart rate [ 39 ] , and so forth. Meanwhile, FER systems appear in literature like the most used solution to recognize student’s affective states in smart classroom.

 
 
 Recently, FER systems are used in several domains like robotics, security, psychology, etc; thus, over time many researchers suggest new performant FER approaches. The framework of FER systems is structured as shown in Figure 2 ; the input of this system is the images from a FER database; those images are pre-processed in a way to match the input size of the network, then, adequate algorithms are used to detect area of the face properly and extract the main features that help the network learn from training data; the final step is to classify results according to the database labels [ 6 ] . Thousands of articles have been written on this subject, but only a few of them have applied FER systems to smart classrooms.

 
 
 Figure 2: Facial expressions recognition system Framework 
 
 
 In Table 2 , we cite FER approaches used in smart classrooms; we classify those approaches into three categories: 
 Handcrafted approaches : are traditional methods of machine learning that consist of manually extracting features. Some examples include edge detection, sharpening, corner detection, histograms, etc. LBP pattern, for instance, is a type of image descriptor used to extract a texture of an image. Then, for classification, an adequate classifier is used; it is also a traditional machine learning algorithm such as: Support Vector Machine (SVM), K-nearest neighbor (KNN), decision tree, etc.
 Deep learning approaches : are new methods based on neural networks for both feature extraction and classification. Learned features are extracted automatically using deep learning algorithms; Convolutional Neural Network (CNN) is the most commonly used for analyzing visual imagery; it can choose the best features to classify the images. 
 Hybrid approaches : combine both algorithms of machine learning and deep learning. They utilize traditional machine-learning algorithms to extract features and neural networks to classify images;

 
 
 Table 2: FER approaches in application to smart classroom 
 
 
 
 Approach | 
 Works | 
 Year | 
 
 
 
 Feature 
 
 extraction 
 | 
 Classification | 
 Accuracy | 

 
 Handcrafted | 
 [ 40 ] | 
 2014 | 
 Gabor features | 
 
 
 
 Support vector 
 
 machine (SVM) 
 | 
 72% | 

 
 [ 10 ] | 
 2015 | 
 ULGBPHS | 
 
 
 
 K-nearest neighbor 
 
 classifier (KNN) 
 | 
 79% | 

 
 Hybrid | 
 [ 41 ] | 
 2019 | 
 LBP-TOP | 
 
 
 
 Deep Neural Network 
 
 (DNN) 
 | 
 85% | 

 
 Deep Learning | 
 [ 42 ] | 
 2019 | 
 
 
 
 Convolutional Neural Network 
 
 (CNN) 
 | 
 70% | 

 
 [ 43 ] | 
 2020 | 
 
 
 
 CNN-1: analyze single face 
 
 expression in single image. 
 
 CNN- 2 analyze multiple faces 
 
 in single image 
 | 
 
 
 
 86% for CNN1 
 
 70% for CNN2 
 | 

 
 [ 38 ] | 
 2020 | 
 
 
 
 CNN based 
 
 on GoogleNet architecture 
 
 with 3 types of Databases 
 | 
 
 
 
 88 % 
 
 79 % 
 
 61% 
 | 

 

 
 

#### Notes and Comments.

 
 In the beginning, approaches (before 2019) employed classical machine learning methods like Gabor filters, LPB, KNN, and SVM to extract features and classify emotions; then gradually, since 2019, authors started integrating neural networks. As shown in Table 2 , [ 41 ] and al utilize a hybrid approach that consists of using LBP-TOP as a descriptor then deep neural network (DNN) for classification which gives a better result (85%) in comparison to both the methods proposed by [ 40 ] [ 10 ] accuracies (respectively 72%, 79% ) and the new deep method suggested by [ 42 ] ; additionally, Ashwin and al. suggest methods with different accuracies improved ( from 61% to 88%) by changing the training data [ 43 ] [ 38 ] ; which point on the importance of choosing an adequate database.

 
 
 
 

### 3.2 FER student’s databases

 
 Over the past years, there was a lack of student facial expressions databases [ 36 ] , and most of the authors use general FER databases like FER2013 and CK+ to train their models [ 44 ] [ 38 ] .
It is essential to have an adequate database to increase the models’ accuracy and get better results [16], because the efficiency of FER models depends mainly on the quality of both databases and FER approaches [19].
Each created database has its characteristics, depending on the classes of expressions, ethnicity of participants, labeling methods, Size, method and angle of image acquisition. In Table 3 , we have gathered students’ expressions databases used in intelligent classrooms.

 
 
 
 Table 3: Student’s FER databases 
 
 
 
 
 Works | 
 Database | 
 
 
 
 Expressions 
 
 Classes 
 | 
 
 
 
 Ethnicity 
 
 gender of 
 
 participants 
 | 
 
 
 
 Method or angle 
 
 of acquisition 
 | 
 Labeling | 
 
 
 
 Database 
 
 content 
 | 

 
 [ 40 ] | 
 Spontaneous | 
 
 
 
 4 classes 
 
 Not engaged 
 
 Nominally engaged 
 
 Engaged in task 
 
 Very engaged 
 | 
 
 
 
 Asian-American 
 
 Caucasian-American 
 
 African-American 
 
 25 female 
 | 
 
 
 
 Pictures are taken 
 
 from an IPad 
 
 camera posed 
 
 in 30 centimeters 
 
 in front of the 
 
 participant’s face 
 
 while playing 
 
 a cognitive game 
 | 
 
 
 
 Labeled by 
 
 students from 
 
 different disciplines: 
 
 computer science, 
 
 cognitive and 
 
 psychological science 
 | 
 
 
 
 N/A 
 
 34 participant 
 | 

 
 [ 10 ] | 
 Spontaneous | 
 
 
 
 5 classes: 
 
 Joviality 
 
 Surprise 
 
 Concentration 
 
 Confusion 
 
 Fatigue 
 | 
 Asian | 
 
 
 
 Acquired using a 
 
 full 1080p HD 
 
 camera configured 
 
 at the front of 
 
 the classroom while 
 
 the student watching 
 
 a 6 minutes video 
 | 
 
 
 
 Participants labeled 
 
 their own pictures 
 | 
 
 
 
 200 images 
 
 23 participants 
 | 

 
 [ 45 ] | 
 Spontaneous | 
 
 
 
 4 Classes : 
 
 Frustration 
 
 Boredom 
 
 Engagement 
 
 Excitement 
 | 
 N/A | 
 
 
 
 Computer webcam 
 
 takes a photograph 
 
 every 5 seconds 
 | 
 
 
 
 Labeled using 
 
 a mobile 
 
 electroencephalo 
 
 -graphy (EEG) 
 
 technology called 
 
 Emotiv Epoc 
 | 
 730 images | 

 
 [ 46 ] | 
 Spontaneous | 
 
 
 
 5 classes: 
 
 Confusion 
 
 Distraction 
 
 Enjoyment 
 
 Neutral 
 
 Fatigue 
 | 
 
 
 
 Chinese 
 
 (29 male and 
 
 53 female) 
 | 
 
 
 
 Acquired using 
 
 Computers cameras 
 | 
 
 
 
 Labeled by 
 
 the participants 
 
 and external coders 
 | 
 
 
 
 1,274 video 
 
 30,184 mages 
 
 82 participants 
 | 

 
 [ 38 ] | 
 
 
 
 Posed and 
 
 spontaneous 
 | 
 
 
 
 14 Classes: 
 
 7 classes of 
 
 Ekman’s basic 
 
 emotions 
 
 3 learning-centered 
 
 emotions 
 
 ( Frustration, 
 
 confusion 
 
 and boredom ) 
 
 Neutral 
 | 
 Indian | 
 
 
 
 Frontal posed 
 
 expressions 
 | 
 
 
 
 Labeled using 
 
 the semi-automatic 
 
 annotation process 
 
 and reviewed 
 
 manually to 
 
 correct wrong 
 
 annotations 
 | 
 4000 | 

 

 
 
 
 
 

## 4 Insights and future work

 
 In this paper, we have surveyed the concept of smart classroom and its technologies, especially FER systems. In the future, we plan to consider the evolution of smart classrooms in this critical period of the pandemic.

 
 
 Today, the pandemic of coronavirus is causing a global health crisis. During this time, countries around the world have imposed restrictions on social distancing, masking, and other aspects of public life. The lockdowns in response to the spread of the virus are significantly impacting educational systems. As a result, governments are striving to maintain continuity of learning and are proposing distance learning as a suitable interim solution, but unfortunately, not all students around the globe have access to digital learning resources. In our future work, we propose a model for an intelligent physical classroom that respects the restrictions of Covid-19. We base our model on two propositions; as shown in Figure 3 :

 
 
 Proposition 1 : we propose a system that automatically detects whether the students in the classroom are wearing their masks or not.

 
 
 Proposition 2 : we detect the distances between students and compare them with the allowed distance.

 
 
 If the students do not comply with the restrictions, the system sends a warning to the teacher in real-time.

 
 
 Figure 3: Proposed model for intelligent classroom 
 
 

#### Notes and Comments.

 
 The Proposed intelligent classroom system in Figure 3 consisting of (Model 1, Model 2) artificial neural networks to detect students not wearing masks and to measure distances between learners.
 Wearing face masks strongly confuses facial emotions recognition systems (FER). In our future work, we will study the limitations of these systems, used in today’s smart classrooms, as well as the possibility of predicting emotions in a student face wearing a mask; since it is possible, for example, to predict students’ mind wandering and the level of their engagement based only on their gaze [ 31 ] [ 32 ] .

 
 
 
 

## 5 Conclusion

 
 Smart classroom is not a new concept, but over the years it has known many changes through the integration of various technologies.
Researchers have transformed the classroom from a simple physical space gathering learners and their instructors to an emotionally aware environment that can interact with students and help them learn efficiently. Numerous technologies have revolutionized the evolution of digital classrooms, and FER systems are considered the most innovative.
The authors have adapted FER ’s systems for use in smart classrooms using approaches with high accuracies and personalized databases.
 The evolution of these interactive classrooms can be considered to aid in teaching during the restrictions of the COVID 19 pandemic. It can also be adapted for teaching students with special needs or intellectual disabilities to facilitate their interaction with their teachers.

 
 
 

## 6 Acknowledgement

 
 Authors would like to thank the National Center for Scientific and Technical
Research (CNRST) for supporting and funding this research.

 
 
 

## References

 
 
 (1) 
 
Zhang, Y., Li, X., Zhu, L., Dong, X., Hao, Q.: What is a smart classroom? a
literature review. In: Perspectives on Rethinking and Reforming Education,
pp. 25–40. Springer Singapore (2019)

 

 
 (2) 
 
Williamson, B.: Decoding classdojo: Psycho-policy, social-emotional learning
and persuasive educational technologies. Learning, Media and Technology
42(4), 440–453 (2017)

 

 
 (3) 
 
Kim, Y., Soyata, T., Behnagh, R.F.: Towards emotionally aware AI smart
classroom: Current issues and directions for engineering and education.
IEEE Access 6, 5308–5331 (2018)

 

 
 (4) 
 
Leony, D., Muñoz-Merino, P.J., Pardo, A., Kloos, C.D.: Provision of
awareness of learners’ emotions through visualizations in a computer
interaction-based environment. Expert Systems with Applications 40(13),
5093–5100 (2013)

 

 
 (5) 
 
Yadegaridehkordi, E., Noor, N.F.B.M., Ayub, M.N.B., Affal, H.B., Hussin, N.B.:
Affective computing in education: A systematic review and future research.
Computers Education 142, 103649 (2019)

 

 
 (6) 
 
Kas, M., merabet, Y.E., Ruichek, Y., Messoussi, R.: New framework for
person-independent facial expression recognition combining textural and shape
analysis through new feature extraction approach. Information Sciences 549,
200–220 (mar 2021)

 

 
 (7) 
 
Kwet, M., Prinsloo, P.: The ‘smart’classroom: a new frontier in the age of
the smart university. Teaching in Higher Education 25(4), 510–526 (2020)

 

 
 (8) 
 
Saini, M.K., Goel, N.: How smart are smart classrooms? a review of smart
classroom technologies. ACM Computing Surveys 52(6), 1–28 (nov 2020)

 

 
 (9) 
 
Miraoui, M.: A context-aware smart classroom for enhanced learning environment.
International Journal on Smart Sensing and Intelligent Systems 11(1), 1–8
(2018)

 

 
 (10) 
 
Tang, C., Xu, P., Luo, Z., Zhao, G., Zou, T.: Automatic facial expression
analysis of students in teaching environments. In: Biometric Recognition, pp.
439–447. Springer International Publishing (2015)

 

 
 (11) 
 
Ashwin, T.S., Guddeti, R.M.R.: Unobtrusive behavioral analysis of students in
classroom environment using non-verbal cues. IEEE Access 7,
150693–150709 (2019)

 

 
 (12) 
 
Pacheco, A., Cano, P., Flores, E., Trujillo, E., Marquez, P.: A smart classroom
based on deep learning and osmotic IoT computing. In: 2018 Congreso
Internacional de Innovación y Tendencias en Ingeniería
(CONIITI). IEEE (oct 2018)

 

 
 (13) 
 
Revathi, R., Suganya, M., NR, G.M., et al.: Iot based cloud integrated smart
classroom for smart and a sustainable campus. Procedia Computer Science 172,
77–81 (2020)

 

 
 (14) 
 
J, F.B., R, R., M, S., R, G.M.N.: IoT based cloud integrated smart classroom
for smart and a sustainable campus. Procedia Computer Science 172, 77–81
(2020)

 

 
 (15) 
 
Kapoor, A., Burleson, W., Picard, R.W.: Automatic prediction of frustration.
International Journal of Human-Computer Studies 65(8), 724–736 (aug 2007)

 

 
 (16) 
 
Kapoor, A., Picard, R.W.: Multimodal affect recognition in learning
environments. In: Proceedings of the 13th annual ACM international
conference on Multimedia - MULTIMEDIA '05. ACM Press
(2005)

 

 
 (17) 
 
Khosravi, S., Bailey, S.G., Parvizi, H., Ghannam, R.: Learning enhancement in
higher education with wearable technology. arXiv preprint arXiv:2111.07365
(2021)

 

 
 (18) 
 
Declaration, I.: Sdg4—education 2030 framework for action (2016)

 

 
 (19) 
 
Lant, C.L., Lawson, M.J.: Interactive whiteboard use and student engagement.
In: Publishing Higher Degree Research, pp. 33–42. SensePublishers (2016)

 

 
 (20) 
 
Glover, D., Miller, D., Averis, D., Door, V.: The interactive whiteboard: a
literature survey. Technology, Pedagogy and Education 14(2), 155–170 (jul
2005)

 

 
 (21) 
 
D., H., Salih, N., Al, A., Al-Sadawi, B., Alsharqi, H.: Attendance and
information system using RFID and web-based application for academic
sector. International Journal of Advanced Computer Science and Applications
9(1) (2018)

 

 
 (22) 
 
Kassim, M., Mazlan, H., Zaini, N., Salleh, M.K.: Web-based student attendance
system using RFID technology. In: 2012 IEEE Control and System Graduate
Research Colloquium. IEEE (jul 2012)

 

 
 (23) 
 
Alimisis, D., Moro, M., Menegatti, E. (eds.): Educational Robotics in the
Makers Era. Springer International Publishing (2017)

 

 
 (24) 
 
Timms, M.J.: Letting artificial intelligence in education out of the box:
Educational cobots and smart classrooms. International Journal of Artificial
Intelligence in Education 26(2), 701–712 (jan 2016)

 

 
 (25) 
 
Akçayır, M., Akçayır, G., Pektaş, H.M., Ocak, M.A.:
Augmented reality in science laboratories: The effects of augmented reality
on university students’ laboratory skills and attitudes toward science
laboratories. Computers in Human Behavior 57, 334–342 (2016)

 

 
 (26) 
 
Thomas, P., David, W.: Augmented reality: An application of heads-up display
technology to manual manufacturing processes. In: Hawaii international
conference on system sciences. pp. 659–669 (1992)

 

 
 (27) 
 
Chen, P., Liu, X., Cheng, W., Huang, R.: A review of using augmented reality in
education from 2011 to 2016. In: Innovations in Smart Learning, pp. 13–18.
Springer Singapore (sep 2016)

 

 
 (28) 
 
Shaping Future Schools with Digital Technology. Springer Singapore (Sep 2019),
 https://www.ebook.de/de/product/37068985/shaping˙future˙schools˙with˙digital˙technology.html 

 

 
 (29) 
 
Fies, C., Marshall, J.: Classroom response systems: A review of the literature.
Journal of Science Education and Technology 15(1), 101–109 (mar 2006)

 

 
 (30) 
 
Martyn, M.: Clickers in the classroom: An active learning approach. Educause
quarterly 30(2),  71 (2007)

 

 
 (31) 
 
Hutt, S., Mills, C., Bosch, N., Krasich, K., Brockmole, J.,
D'Mello, S.: ”out of the fr-eye-ing pan”. In: Proceedings of
the 25th Conference on User Modeling, Adaptation and Personalization. ACM
(jul 2017)

 

 
 (32) 
 
Hutt, S., Krasich, K., Mills, C., Bosch, N., White, S., Brockmole, J.R.,
D’Mello, S.K.: Automated gaze-based mind wandering detection during
computerized learning in classrooms. User Modeling and User-Adapted
Interaction 29(4), 821–867 (jun 2019)

 

 
 (33) 
 
Courts, B., Tucker, J.: Using technology to create a dynamic classroom
experience. Journal of College Teaching Learning (TLC) 9(2),
121–128 (mar 2012)

 

 
 (34) 
 
Conde, M.Á., García-Peñalvo, F.J., Rodríguez-Conde,
M.J., Alier, M., Casany, M.J., Piguillem, J.: An evolving learning management
system for new educational environments using 2.0 tools. Interactive Learning
Environments 22(2), 188–204 (dec 2012)

 

 
 (35) 
 
Alhazmi, A.K., Rahman, A.A.: Why lms failed to support student learning in
higher education institutions. In: 2012 IEEE symposium on e-learning,
e-management and e-services. pp. 1–5. IEEE (2012)

 

 
 (36) 
 
Gligorić, N., Uzelac, A., Krco, S.: Smart classroom: real-time feedback on
lecture quality. In: 2012 IEEE International Conference on Pervasive
Computing and Communications Workshops. pp. 391–394. IEEE (2012)

 

 
 (37) 
 
Kumar, J., kumar, J.: Affective modelling of users in HCI using EEG.
Procedia Computer Science 84, 107–114 (2016)

 

 
 (38) 
 
T.S., A., Guddeti, R.M.R.: Affective database for e-learning and classroom
environments using indian students’ faces, hand gestures and body postures.
Future Generation Computer Systems 108, 334–348 (jul 2020)

 

 
 (39) 
 
Monkaresi, H., Bosch, N., Calvo, R.A., D'Mello, S.K.:
Automated detection of engagement using video-based estimation of facial
expressions and heart rate. IEEE Transactions on Affective Computing 8(1),
15–28 (jan 2017)

 

 
 (40) 
 
Whitehill, J., Serpell, Z., Lin, Y.C., Foster, A., Movellan, J.R.: The faces of
engagement: Automatic recognition of student engagementfrom facial
expressions. IEEE Transactions on Affective Computing 5(1), 86–98 (jan
2014)

 

 
 (41) 
 
Kaur, A., Mustafa, A., Mehta, L., Dhall, A.: Prediction and localization of
student engagement in the wild. In: 2018 Digital Image Computing: Techniques
and Applications (DICTA). IEEE (dec 2018)

 

 
 (42) 
 
Lasri, I., Solh, A.R., Belkacemi, M.E.: Facial emotion recognition of students
using convolutional neural network. In: 2019 Third International Conference
on Intelligent Computing in Data Sciences (ICDS). IEEE (oct 2019)

 

 
 (43) 
 
S., A.T., Guddeti, R.M.R.: Automatic detection of students’ affective states in
classroom environment using hybrid convolutional neural networks. Education
and Information Technologies 25(2), 1387–1415 (oct 2019)

 

 
 (44) 
 
Ashwin, T.S., Guddeti, R.M.R.: Impact of inquiry interventions on students in
e-learning and classroom environments using affective computing framework.
User Modeling and User-Adapted Interaction 30(5), 759–801 (jan 2020)

 

 
 (45) 
 
Zatarain-Cabada, R., Barron-Estrada, M.L., Gonzalez-Hernandez, F.,
Rodriguez-Rangel, H.: Building a face expression recognizer and a face
expression database for an intelligent tutoring system. In: 2017 IEEE 17th
International Conference on Advanced Learning Technologies (ICALT). IEEE
(jul 2017)

 

 
 (46) 
 
Bian, C., Zhang, Y., Yang, F., Bi, W., Lu, W.: Spontaneous facial expression
database for academic emotion inference in online learning. IET Computer
Vision 13(3), 329–337 (mar 2019)