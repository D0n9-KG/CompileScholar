A Comprehensive Survey of Machine Learning Based Localization with Wireless Signals 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2012.11171v1 [eess.SY] 21 Dec 2020 
 
 

# A Comprehensive Survey of Machine Learning Based Localization with Wireless Signals

 Thanks: D. Burghal, A. T. Ravi, A. F. Molisch are at the Ming Hsieh Department of Electrical Engineering, University of Southern California, Los Angeles, CA 90089, USA. (e-mail:{burghal,telagima,molisch}@usc.edu. Thanks: V. Rao was with the University of Southern California, Los Angeles, CA 90089 USA. He is now with Amazon.com Services, Inc., Seattle, 98109 USA (e-mail: varunrao.vr41@gmail.com). Thanks: A. A. Alghafis is with the Communication and Information Technology Research Institute, King Abdulaziz City for Science and Technology, Riyadh 11442, Saudi Arabia (e-mail: alghafis@kacst.edu.sa). 

 
 
 Daoud Burghal
 
    
 Ashwin T. Ravi
 
    
 Varun Rao
 
    
 Abdullah A. Alghafis
 
    
 Andreas F. Molisch
 
    
 Fellow, IEEE 
 

 Abstract 
 
 The last few decades have witnessed a growing interest in location-based services. Using localization systems based on Radio Frequency (RF) signals has proven its efficacy for both indoor and outdoor applications. However, challenges remain with respect to both complexity and accuracy of such systems. Machine Learning (ML) is one of the most promising methods for mitigating these problems, as ML (especially deep learning) offers powerful practical data-driven tools that can be integrated into localization systems. In this paper, we provide a comprehensive survey of ML-based localization solutions that use RF signals. The survey spans different aspects, ranging from the system architectures, to the input features, the ML methods, and the datasets.

 
 A main point of the paper is the interaction between the domain knowledge arising from the physics of localization systems, and the various ML approaches. Besides the ML methods, the utilized input features play a major role in shaping the localization solution; we present a detailed discussion of the different features and what could influence them, be it the underlying wireless technology or standards or the preprocessing techniques. A detailed discussion is dedicated to the different ML methods that have been applied to localization problems, discussing the underlying problem and the solution structure. Furthermore, we summarize the different ways the datasets were acquired, and then list the publicly available ones. Overall, the survey categorizes and partly summarizes insights from almost 400 papers in this field.

 
 This survey is self-contained, as we provide a concise review of the main ML and wireless propagation concepts, which shall help the researchers in either field navigate through the surveyed solutions, and suggested open problems.

 
 
 

## I Introduction 

 
 Location information is becoming a cornerstone for many applications, e.g., smart transportation systems, assisted driving services, public safety, location-based advertising, individualized location-based tourist information, and many others. Part of this success can be attributed to the wide availability of efficient localization with Global Navigation Satellite Systems (GNSSs), in particular GPS (Global Positioning System). As a matter of fact, in environments where GNSSs are available, the localization problem can be considered as mostly solved. However, localization with a GNSS is based on trilateration, using radio frequency (RF) signals, which requires free line of sight (LOS) to at least 3 satellites. As a result, in indoor environments, dense urban environments (e.g., street canyons), underground (e.g., mines), and other environments or scenarios, localization is challenging and has many open problems, even after many years of research.

 
 
 In the last two decades, efforts were made to use light, magnetic field, acoustic signals, images, and inertial sensor readings for localization. However, RF-based localization systems have continued to attract most of the research interest due to the wide availability of wireless systems and technologies, and the attractive properties of RF signals [ 1 ] .

 
 

### I-A Localization Methods and Challenges 

 
 RF based localization methods can be roughly categorised into four main classes:

 
 • 
 
 Trilateration and triangulation methods . The system uses one or more observed quantities to infer the distance or the angles between the target (i.e., a wireless node whose position we wish to determine), and anchors (i.e., nodes with known location). It then applies simple geometric calculation to infer the coordinates of the target. GNSS, and, more generally, time-of-arrival (ToA) -based trilateration [ 2 ] are the most important example of this category.

 

 • 
 
 Proximity . In these solutions, the goal is to identify whether a target is within a certain range from an infrastructure node (e.g., a sensing devices).

 

 • 
 
 Fingerprint matching . In this type of solutions, the system relies on the availability of a database that contains the fingerprints, i.e., observations of characteristics of signals at a set of locations; when localization of a target is required, the system uses either statistical or deterministic methods to match the observed signal to one of the priori stored fingerprints to infer the approximate user coordinates. The construction of the database is usually done ”offline”, e.g., when the system is initially deployed.

 

 • 
 
 Direct methods . In these solutions, the system estimates the coordinates directly without estimating latent quantities such as distances and angles. A possible way is through a set of assumptions about the joint relation between the coordinates and wireless channels, e.g., statistical models.

 

 
 
 
 None of these approaches can be considered as universally ”the best”, as the success of any method is contingent on the availability of the required data and technology, level of complexity, and the properties of the environment or the setup. For instance, ToA-based trilateration works best for wideband systems that have precise synchronization between the anchors, and operating in environments where pure LOS propagation to at least three anchors is present; triangulation requires antenna arrays and also environments with LOS or quasi-LOS propagation. Proximity-based methods, in applications where the relative range is enough, may require additional infrastructure. Finger printing techniques are mainly constrained by the difficulty and expense of building up the database (also known as radio-map); furthermore establishing a framework for robust and accurate identification of the similarity between the observed channel and stored fingerprint is not trivial.

 
 
 The attempt to use model-based direct methods relies on the accuracy of the underlying assumptions, e.g., the statistical distribution of the signal, which are usually constrained by requirements for analytical tractability and thus could deviate from actual environmental properties. Note that triangulation/trilateration are based on a physical model of the wireless propagation. This model relates the position of the device to some intermediate characteristics, such as signal runtime, that in turn can be derived from the received signals. The advantage of such an approach is a clear physical relationship between the measurements and the solutions. However, model-based methods become less and less reliable in the presence of effects that are not included in the underlying models - be it because they cannot be measured realistically (such as a calibrated antenna pattern for every single handset), because their inclusion in the model would be too complicated (such as non-Gaussian noise statistics), or because they would make computations too time-consuming for real-time implementation. Important information might be lost due to the necessity of simplified models.

 
 
 

### I-B A Case for Machine Learning 

 
 To address some of the aforementioned limitations, recent years have seen increased interest in utilizing Machine Learning (ML) based localization. ML has emerged as a powerful framework to solve many challenging practical problems in a wide variety of areas, ranging from image recognition, to translation between different languages, to scheduling of wireless transmissions, to autonomous driving. Fundamentally, ML uses real-world data to train an ML solution to capture the complex relations between the input data (features) and the output values (labels).

 
 
 The labels can be limited to discrete values that represent different classes, such as classifying an observed image to be one of N N objects, e.g., dog vs. cat, in which case the problem is referred to as classification problem . Alternatively, the labels could be a range of continuous values, such as the price of a stock; this problem is referred to as regression problem . Note that identifying a location can be formulated as either a classification problem (e.g., finding a point on a finite grid that best fits the observed input data) or a regression problem (finding the - continuous - coordinates of the device).

 
 
 Figure 1: Sections and main subsections of the paper. 
 
 
 ML-based localization can be used in any of the introduced localization methods. Direct methods are a natural choice for an end-to-end ML-based solution, where observations are fed to the ML solution to produce the location. However, ML can also be integrated with any of the other localization methods. For instance, ML can be used to improve the range estimates in trilateration and proximity methods. They can be used as robust matching techniques in fingerprinting methods. The range of applications extends to more complicated systems, e.g., when the available data are constrained or when observations are complex-valued. This has ignited rapidly growing interest in ML-based localization, which can be attributed to the following.

 
 • 
 
 Recent success of ML in many fields, in particular image recognition and speech and language processing, proving that highly complex problems that have long stymied deterministic approaches can be successfully tacked with ML, and allowing the solutions to be not limited by underlying model assumption.

 

 • 
 
 The ability of ML to utilize large sets of available data; the data can also be from heterogeneous sources, which are usually difficult to utilize with deterministic solutions.

 

 • 
 
 A number of localization problems can be naturally formulated as ML problems.

 

 • 
 
 The availability of efficient software and hardware solutions that are customized to accelerate ML algorithms. For instance, it is anticipated that by 2022 about 80 percent of the shipped smartphones have on-device Artificial Intelligence (AI) capabilities [ 3 ] .

 

 
 At the same time, ML approaches to localization face major challenges

 
 • 
 
 Availability of training data: large sets of training data are sometimes required for number of ML algorithms.

 

 • 
 
 Robustness: it is uncertain how robust ML solutions are to both small and large changes in the measurement devices and the environment, and how robust to interference. For example, can the presence/absence of a truck near a receiver completely throw off the location estimate? What is the impact of RF signal interference in cognitive and ad-hoc networks? 1 1 
 1 
 
 
 
 While it is possible that localization systems may face malign interference, such as jamming, this is rare in most civil applications. Such an issue could occur in other applications such as radars [ 4 ] , which is beyond the scope of this survey. Such interference could lead to corruption of (part of) the training data or the feature measurements of the device.

 

 • 
 
 Training and computational complexity: are the algorithms simple enough to be trained and performed in real-time? This is particularly important if the localization should be done by the user device (an approach that is desirable for privacy reasons).

 

 • 
 
 Feature selection: what are the signal characteristics that should be used as the input for the ML algorithm.

 

 
 
 
 

### I-C Contributions 

 
 The anticipated accelerated research in ML-based localization calls for a comprehensive review of the fundamental aspects of ML-based localization solutions and relevant problems, as well as a summary of what has been done, and what still remains to do. The current paper aims to provide exactly such a survey.

 
 
 In the literature, there have been excellent surveys of localization techniques. However, most of the available surveys are presented with the focus on solutions that are specific to certain environments [ 5 , 6 , 7 , 8 ] , technologies or/and standards [ 9 , 10 , 11 , 8 , 12 ] , applications [ 13 , 14 , 15 ] , localization methods [ 11 , 16 ] , type of used signal [ 17 , 18 , 19 , 20 ] . Furthermore, most of these surveys, as well as the extensive handbook [ 1 ] , concentrate on conventional (i.e., non-ML) approaches. A few provide explicit yet very limited discussion of some ML-based solutions [ 13 , 11 , 6 , 18 , 5 ] . In one of the recent papers [ 21 ] , the authors provide a survey of a dimensionality reduction based localization, which is usually considered an ML technique (see Sec. II-A ). There are recent papers that survey ML-based solutions for non RF based localization, e.g., [ 22 ] . Different from all the above, in this paper, we provide a self-contained and comprehensive survey of ML-based localization solutions that utilize RF signals.

 
 
 In particular, the contributions of this paper are as follows.

 
 • 
 
 We provide comprehensive review of the ML-based localization systems based on different aspects:

 
 – 
 
 Localization goal.

 

 – 
 
 System/target engagement.

 

 – 
 
 The used RF features.

 

 – 
 
 The used wireless technologies and standards along with their impact on the features.

 

 – 
 
 Details of the ML solutions based on the ML framework and system setup. The presentation is carried out from the ML perspective, where different ML solutions and methods may differ in the underlying assumption about the type and the availability of the data (e.g., supervised, unsupervised, semi-supervised, transfer learning, etc., see Sec. II-A for definitions).

 

 
 

 • 
 
 A survey of key papers in various categories. We describe salient features that occur as a common theme in multiple papers, and provide tables that categorize related papers.

 

 • 
 
 We highlight the current challenges that face ML-based localization solutions, and propose a number of interesting directions for future research.

 

 • 
 
 Summaries of publicly available datasets and how data have been collected, are provided.

 

 • 
 
 We provide concise reviews of fundamental ML techniques and wireless channels and systems, to make the paper self-contained and accessible to researchers in both the wireless and the ML communities.

 

 
 
 
 The majority of the surveyed papers are published after 2010, with the vast majority in the last couple of years, when the ML-based localization witnessed an increased interest due to the success of Deep Learning (DL) -based methods. Nevertheless, for completeness, the survey also includes earlier works, as the interest in ML-based localization started about two decades ago (see Sec. IV-F for relevant discussions).

 
 
 

### I-D Paper Structure 

 
 The structure of this survey is as follows. In the first subsection of Sec. II , we provide a concise yet comprehensive overview of ML, which covers many important concepts and aspect of ML and DL, in the second subsection we review relevant concepts of wireless channels and systems. This section is designed to provide an introduction for the researchers who might appreciate a tutorial or refresher in any of the two subjects, and also serves to establish some notation. We review the ultimate goals of the localization solution in Sec. III ; we also discuss the role of the target object in the localization system. Sec. IV is dedicated to wireless features, where we present the basic features used, their relation to wireless technologies, and the common wireless standards; we also discuss some of the side information that has been used to improve the localization. Next, in Sec. V , we summarize the framework based on the availability of the data during the training phase and the learning framework. In each category, we present a number of the proposed ML structures, where we review the used features, input-output relations, and solution details. A major part of Sec. V is dedicated to DL solutions, as those are the subject of many recent advances in ML-based localization. Since the ML solutions depend on the used data, in Sec. VI , we review the different types of datasets that were used to train and evaluate the models, and provide a list of the publicly available datasets for localization. In Sec. VII , we review different challenges that could face ML-based localization solutions. We also provide some suggested open and interesting research directions. Throughout the paper we provide extensive citations. Finally, in Sec. VIII , we provide some concluding remarks followed by a table of the used acronyms (table VIII ). Fig. 1 displays the main subsections of the survey.

 
 
 
 

## II Preliminaries 

 
 In this section we review some fundamental concepts of ML and of wireless communication, in particular wireless propagation channels. Since this review aims at people with different backgrounds, this covers some introductory concepts. It also helps the reader in identifying the novelty of some of the reviewed papers in the later sections, and assess the challenges of using RF signals for localization and ML techniques. The reader may skim or skip the subsections already familiar to them without losing readability of the later sections.

 
 

### II-A Machine Learning 

 
 ML provides a set of tools and frameworks that enable the computer to utilize available data 𝒟 \mathcal{D} to perform a certain task, or detect patterns, without programming it with task- specific procedures. The data usually include observable attributes that the system uses to make decisions. ML can be classified into different categories depending on what and when data is available, what task we try to achieve, and the complexity constraints on the models.
We can initially distinguish two main classes:

 
 • 
 
 In supervised learning , the system has a training dataset 𝒟 \mathcal{D} to learn the mapping between the right decisions (labels or tags) y y and the observed features 𝒙 ∈ 𝐑 d \boldsymbol{x}\in\mathbf{R}^{d} . 𝒟 \mathcal{D} consists of N N examples (or data points) with features-label pairs, i.e., 𝒟 = { 𝒙 i , y i } i = 1 N \mathcal{D}=\{\boldsymbol{x}^{i},y^{i}\}_{i=1}^{N} .

 

 • 
 
 In contrast, in unsupervised learning , the collected dataset has no labels, i.e., 𝒟 = { 𝒙 i } i = 1 N \mathcal{D}=\{\boldsymbol{x}^{i}\}_{i=1}^{N} .

 

 
 The two approaches usually differ in the task. In the following we start with standard supervised learning to introduce many ML concepts. We then discuss unsupervised learning, followed by a brief introduction to other learning classes. Next, we discuss DL and briefly review some of the available frameworks. Other details and categories are provided later in the survey as needed.

 
 

#### II-A 1 Supervised Learning

 
 Supervised learning uses labeled training data, i.e., examples annotated with correct decisions, to teach the system how to make decision when it observes data outside the set of examples. More formally, assuming that the input output relation is given as follows:

 

 
 | 
 y = f ⁡ ( 𝒙 ) + ϵ , \displaystyle y=f(\boldsymbol{x})+\epsilon\penalty\ \penalty\ , | 
 | 
 (1) | 
 

 where the vector of d d features 𝒙 ∈ 𝐑 d \boldsymbol{x}\in\mathbf{R}^{d} is the observable attributes, y y is the associated decision (usually referred to as the label), f f is an unknown mapping function, and ϵ \epsilon is the noise that encompasses any unobserved factors or intrinsic randomness. When y y takes categorical values, we refer to the problem as a classification problem , e.g., y y is one of four directions { east , west , north , south } \{\rm east,west,north,south\} . When y y takes continuous values, the problem is referred to as a regression problem , e.g, y ∈ 𝐑 y\in\mathbf{R} is the distance to an object. Then the goal of a discriminative ML is to learn the underlying mapping f f , such that y ^ ≜ f ^ ​ ( 𝒙 , Ω ) ≈ y \hat{y}\triangleq\hat{f}(\boldsymbol{x};\Omega)\approx y , where f ^ \hat{f} is the learned mapping, and Ω \Omega are the mapping parameters. Alternatively it can be viewed as learning the conditional distribution p ⁡ ( y | 𝒙 ) p(y|\boldsymbol{x}) using the model p ⁡ ( y | 𝒙 ; Ω ) p(y|\boldsymbol{x};\Omega) , here 𝒙 \boldsymbol{x} values can be assumed to be generated from a joint probability distribution p ⁡ ( 𝒙 ) p(\boldsymbol{x}) . The probability notation is useful as typically the features are random, and/or the mapping is uncertain due to the noise and other hidden factors. The relation between the two, the learned model f ^ \hat{f} and distribution p ⁡ ( y | 𝒙 ; Ω ) p(y|\boldsymbol{x};\Omega) , may be given by [ 23 ] 

 

 
 | 
 y ^ = f ^ ​ ( 𝒙 , Ω ) = argmax y ​ p ​ ( y | 𝒙 ; Ω ) . \hat{y}=\hat{f}(\boldsymbol{x};\Omega)={\rm argmax}_{y}p(y|\boldsymbol{x};\Omega)\penalty\ \penalty\ . | 
 | 
 

 
 
 Loss function: 
To train the model, the ML algorithm relies on the dataset 𝒟 \mathcal{D} and a loss metric ℒ \mathcal{L} . The loss metric measures the deviation of the learned mapping y ^ \hat{y} from the true labels y y , i.e., ℒ ⁡ ( y , y ^ ) \mathcal{L}(y,\hat{y}) . A popular loss function is mean squared error (MSE):

 
 
 

 
 | 
 ℒ MSE ​ ( y , y ^ ) = 𝔼 ⁡ [ | y − f ^ ​ ( 𝒙 , Ω ) | 2 ] ≈ 1 N ​ ∑ i N | y i − f ^ ​ ( 𝒙 i , Ω ) | 2 \mathcal{L}_{\rm MSE}(y,\hat{y})=\mathbb{E}[|y-\hat{f}(\boldsymbol{x};\Omega)|^{2}]\approx\frac{1}{N}\sum^{N}_{i}|y^{i}-\hat{f}(\boldsymbol{x}^{i};\Omega)|^{2} | 
 | 
 

 where 𝔼 ⁡ [ ] \mathbb{E}[] is the expectation operator over the ensemble of noise and 𝒙 \boldsymbol{x} realizations, which may be approximated using the empirical mean of the square error over the N N examples. From a probabilistic perspective, we can use Kullback–Leibler (KL) divergence to measure the dissimilarity between p ⁡ ( y | 𝒙 ) p(y|\boldsymbol{x}) (from of the dataset) and the model p ⁡ ( y | 𝒙 ; Ω ) p(y|\boldsymbol{x};\Omega) , in this case

 

 
 | 
 ℒ K ​ L ​ ( p ⁡ ( y | 𝒙 ) , p ⁡ ( y | 𝒙 ; Ω ) ) \displaystyle\mathcal{L}_{KL}(p(y|\boldsymbol{x}),p(y|\boldsymbol{x};\Omega)) | 
 = 𝔼 ⁡ [ log ⁡ p ⁡ ( y | 𝒙 ) − log ⁡ p ⁡ ( y | 𝒙 ; Ω ) ] \displaystyle=\mathbb{E}[\log p(y|\boldsymbol{x})-\log p(y|\boldsymbol{x};\Omega)] | 
 | 
 (2) | 
 

 Under certain conditions, a number of loss functions are equivalent. For instance, minimizing the loss function above means that we have to select Ω \Omega such that ℒ K ​ L \mathcal{L}_{KL} is small, which can be shown to be equivalent to minimizing − 𝔼 ⁡ [ log ⁡ p ⁡ ( y | 𝒙 ; Ω ) ] -\mathbb{E}[\log p(y|\boldsymbol{x};\Omega)] , and this is equivalent to maximizing the log-likelihood of the distribution over Ω \Omega [ 24 ] . Furthermore, when the noise in ( 1 ) has Gaussian distribution, maximizing the likelihood function (i.e., maximizing log ⁡ p ⁡ ( y | 𝒙 ; Ω ) \log p(y|\boldsymbol{x};\Omega) ) results in a metric equivalent to the MSE [ 24 ] . Finally, we note that minimizing the KL divergence is equivalent to the popular cross-entropy loss function that is widely used in classification problems.

 
 
 Finally, the structure of f ^ \hat{f} (or p ( ; Ω ) p(;\Omega) ) depends on our chosen model; it is usually controlled by the set of parameters Ω \Omega . The model and the possible set of parameters define the hypothesis set ℋ \mathcal{H} that we hope that f f belongs to. The choice of the model is usually made a priori based on domain knowledge, i.e., experience or insights into the physics of the underlying problem. When the size of the model’s parameters is fixed, the model is referred to as a parametric model, otherwise it is non-parametric .

 
 
 Examples: 
To demonstrate the above, let us consider the popular linear regression problem. In such problem we use the data to approximate the scalar output y y as follow:

 

 
 | 
 y ^ = ∑ i = 1 d ω i ​ x i + ω 0 = 𝝎 ⊤ ​ 𝒙 + ω 0 \displaystyle\hat{y}=\sum_{i=1}^{d}\omega_{i}x_{i}+\omega_{0}=\boldsymbol{\omega}^{\top}\boldsymbol{x}+\omega_{0} | 
 | 
 (3) | 
 

 Here the goal of the ML problem is to find the best estimates of the parameters Ω = { 𝝎 , ω 0 } \Omega=\{\boldsymbol{\omega},\omega_{0}\} . When the loss function is the MSE, methods such as least square can be used to estimate the parameters, i.e., ”train” the model.

 
 
 Although the output can take any real number, it can be used for classification, e.g., binary classification based on the sign of y y . However, in that case logistic regression is more appropriate, as it estimates the probability of being in one of two classes. It has the simple formulation

 

 
 | 
 p ⁡ ( y | 𝒙 ; Ω ) = σ ⁡ ( 𝝎 ⊤ ​ 𝒙 + ω 0 ) , \displaystyle p(y|\boldsymbol{x};\Omega)=\sigma(\boldsymbol{\omega}^{\top}\boldsymbol{x}+\omega_{0}), | 
 | 
 (4) | 
 

 
 
 σ ( . ) \sigma(.) is a non-linear function that modifies the output such that it lies between [0,1], thus naturally fits into the probabilistic view; the logistic (sometimes referred to as sigmoid) function is usually used, see Fig. 4 . In this case it is more appropriate to maximize the likelihood function, i.e., p ⁡ ( y | 𝒙 , Ω ) p(y|\boldsymbol{x},\Omega) , with respect to Ω \Omega , which can be shown to be equivalent to minimizing the cross-entropy loss. Different from linear regression, training a logistic regression requires iterative methods such as gradient descent. Both the linear regression and logistic regression are examples of parametric models. An important point to note here is that we can incorporate non-linear features in the linear regression framework (and the logistic regression) by replacing 𝒙 \boldsymbol{x} in ( 3 ) with ϕ ⁡ ( 𝒙 ) \phi(\boldsymbol{x}) , where ϕ ( . ) \phi(.) is, e.g., a polynomial function. This will increase the dimensionality of the features and possibly increase the separability of the data.

 
 
 Another popular model is the K-nearest Neighbor (KNN) model. Different from above, no explicit functional form is assumed, rather, the predicted label of a newly observed data point o o with observations x o x_{o} is based on the labels of the K-nearest neighbors. In a classification problem, with M M different classes, i.e., y ∈ { c 1 , . . , c M } y\in\{c_{1},..,c_{M}\} , the label of the observed point can be given by:

 

 
 | 
 y ^ o = max ⁡ ∑ i ∈ 𝒦 o c ⁡ 𝟙 ​ ( y i = c ) \hat{y}_{o}=\max_{c}\sum_{i\in\mathcal{K}_{o}}\mathbbm{1}(y_{i}=c) | 
 | 
 

 where 𝒦 o \mathcal{K}_{o} is the set of K K neighbors of o o , 𝟙 ​ ( y i = c ) \mathbbm{1}(y_{i}=c) is equal to one if the label of the i th i^{\rm th} neighbor is c c . The choice of K K and what defines a ”neighbor”, i.e., members of 𝒦 o \mathcal{K}_{o} , are design parameter choices that can be set using the data and the prior knowledge. For instance, Euclidean distance can be used to set the distance to points and thus select the K K nearest neighbors. Note that KNN is an example of a non-parametric model. It also belongs to the class to instance-based learning as it derives the labels of the new observations by comparing them with the training instances.

 
 
 Generalization and over-fitting: In supervised learning the goal is to label an unobserved data point, meaning we are interested in minimizing the loss function of the unobserved data. To do so we usually use the observed dataset to estimate the performance over the unobserved data, and thus rely on the generalization capability of the model. 2 2 
 2 
 
 
 
 This statement is a key difference that differentiates ML from conventional optimization problems [ 24 ] . , 3 3 
 3 
 
 
 
 Note that for the system to generalize well beyond the set of the observed examples, smoothness is assumed, such that we can anticipate similar decisions for similar inputs. This poses challenges when we train the model. As per the discussion above, the goal is to minimize the loss function. Yet this may result in overfitting , where we use all the degrees of freedom in the model to fit all the variations in observed values (even the noise). In this case the trained model will have large error values when applied to unobserved data, i.e., results in a large variance. This usually occurs when the model has large degrees of freedom or the dataset is very small. On the other hand, when the model is relatively simple, i.e., cannot capture the true f f , the estimate f ^ \hat{f} will be biased, as we will always have non zero error with any choice Ω \Omega for f ^ ( . ; Ω ) \hat{f}(.;\Omega) . This results in what is known as ”variance-bias trade” off, which can be mapped to the size of the hypothesis set above. Note that ideally we would like to have f ∈ ℋ f\in\mathcal{H} , however, since it is unknown it is quite likely to fall into one of the two extremes above. 4 4 
 4 
 
 
 
 In learning theory, the structure of the hypothesis set ℋ \mathcal{H} can be used to assess generalization error by bounding the deviation of the generalization error from the training error. 

 
 
 Typically, over-fitting is more challenging than under-fitting, since the training error could be misleading. There are different ways to combat over-fitting, including weight regularization and validation. Many training problems can be viewed as parameter optimization. Weight regularization is a powerful technique that prevents the optimizer from over-fitting through penalizing model parameters, for instance, the objective function can be

 

 
 | 
 argmin Ω ​ ℒ ​ ( y , f ^ ​ ( x , Ω ) ) + λ ​ r ​ ( Ω ) \displaystyle{\rm argmin}_{\Omega}\mathcal{L}(y,\hat{f}(x;\Omega))+\lambda r(\Omega) | 
 | 
 (5) | 
 

 where r ⁡ ( ) r() is a function of the parameters Ω \Omega that enforces certain properties on the parameters. A common choice is r ⁡ ( Ω ) = ‖ Ω ‖ p r(\Omega)=||\Omega||_{p} , where sub-script p p refers to the p p -norm (usually referred to as L p L_{p} norm) that can prevent the parameters from taking arbitrarily large values to fit outliers (or noise). λ \lambda is a weight value that controls the importance of the regularization. The choice of p p also impacts the structure of the solution, e.g., p = 0 p=0 or p = 1 p=1 enforce sparseness of the model, and thus plays a role in the model selection.

 
 
 Validation is a powerful technique that can be used to combat overfitting and improve the overall performance of the model. Typically we divide the available dataset into disjoint training and testing subsets, where the testing subset is not used in any way to tune or select the model but used to evaluate the performance of the final model. When validation is used, we divide the training dataset into two (or more) subsets: one for training (parameter tuning) and the other for hyperparameter and model selection. For instance, we can generalize f ^ \hat{f} to a polynomial with degree n n ; the choice of the degree can be selected with validation. The proper choice of the hyperparameter λ \lambda in eq. ( 5 ) can be done also through validation. The concept is simple: since noise and outliers are likely to be different in unobserved realizations in the validation dataset, the model is expected to have larger error values, which could reflect the true performance. As a result, during training, when the model starts to fit noise values to drive down the training error, the validation error goes up. There are different validation techniques, a popular one is the K-fold validation [ 25 , 26 ] , where the dataset is divided into K K different subsets. At each time, K − 1 K-1 subsets are used for training and the other one is used for validation. The model structure and the hyper-parameters are chosen based on the validation error.

 
 
 In practice, preprocessing is usually used to improve the training and model robustness. This includes data formatting, normalization and filtering. Each has different impact on the quality of the model. Noise reduction and outlier filtration could reduce the overfitting. Dimensionality reduction and reducing co-linearity make the learning easier [ 25 , 24 ] . Also proper formatting of the output values impact the choice of the algorithm (e.g., representation of the different classes). Feature normalization also speeds up training and allows the model to capture the impact of feature variation correctly.

 
 
 Probabilistic models: 
A supervised learning solution can also aim at learning the joint distribution of the features and the labels, p ⁡ ( y , 𝒙 ) p(y,\boldsymbol{x}) , or the conditional distribution p ⁡ ( 𝒙 | y ) p(\boldsymbol{x}|y) . In this case the model is referred to as generative model. One can then use Bayes rule on these distributions to get to discriminative approach, i.e., p ⁡ ( y | 𝒙 ) p(y|\boldsymbol{x}) , where the goal of the latter is to find the best decision boundary given the data. An example of generative models is a Naive Bayes classifier that assumes conditional independence of the features given the class (the label). The training for generative models can be done using Maximum Likelihood or Maximum A Posteriori (MAP). For Naive Bayes the estimates of p ⁡ ( 𝒙 | y ) p(\boldsymbol{x}|y) can be simply done by empirically calculating the frequency of a certain feature given class type. The class prior p ⁡ ( y ) p(y) is also estimated empirically from the given labeled dataset. Generative models have many advantages, such as their ability to handle missing data, utilize unlabeled data, and impose prior knowledge. However, they are sensitive to pre-processing and make assumptions that may not be valid in reality.

 
 
 In probabilistic models we have the features 𝒙 \boldsymbol{x} , the labels y y and other latent (unobserved) variables 𝒛 \boldsymbol{z} . The inclusion of latent variables adds power to the model as it could capture the true structure of the real world. In general, we can view y y as one of the latent variables; a generative model would then capture p ⁡ ( 𝒙 , 𝒛 ) p(\boldsymbol{x},\boldsymbol{z}) . It is typically difficult to deal with high dimensional distributions as the number of parameters will grow fast with the number of variables, making both the inference and learning process very difficult [ 23 ] . To alleviate some of these issues, certain restrictions on the variable relations can be used, such as conditional independence (CI) (e.g., two variables are CI when a third variable is given), or limiting the dependency by, e.g., Markov models. With hidden variables, Hidden Markov Models (HMM) can be used. One way to represent models with hidden variables is through Mixture Models, where the distribution can be (roughly) viewed as a mixture of a number of base distributions [ 23 ] . A popular example is the Gaussian Mixture Model (GMM), where the base models are assumed to be Gaussian. With hidden variables, the problem of Maximum Likelihood (and MAP) is usually not convex. In that case, an iterative method such as Expectation Maximization (EM) can be used to arrive at good solutions.

 
 
 An alternative way to capture the variable dependency is through graphical models : Using graphs, the variables are represented by vertices; an edge between two vertices indicates a direct relation between the two variables, a missing link indicates CI [ 27 , 23 ] . The edges can be either directed or undirected, both have their pros and cons that are beyond the scope of this overview. Using either representation allow the use of many graph theoretic algorithms to simplify the learning and inference. Note that, in general, training such models requires sampling from the underlining distribution; methods such as Markov Chain Monte Carlo Methods and Gibbs sampling are usually used.

 
 
 One method to simplify the training of the probabilistic models is to approximate p ⁡ ( 𝒛 | 𝒙 ; Ω ) p(\boldsymbol{z}|\boldsymbol{x};\Omega) with a q ⁡ ( 𝒛 | 𝒙 ; Φ ) q(\boldsymbol{z}|\boldsymbol{x};\Phi) that has a tractable form, then maximize a lower bound that captures the relation between the two (along with the marginal distribution over 𝒙 \boldsymbol{x} ). This is usually referred to as variational inference , and the lower bound is called Evidence Lower Bound (ELBO). Relevant examples are covered in the DL sections below.

 
 
 

#### II-A 2 Unsupervised Learning 

 
 In unsupervised learning the dataset has only unlabeled data, 𝒟 = { 𝒙 i } i = 1 N \mathcal{D}=\{\boldsymbol{x}^{i}\}_{i=1}^{N} . The goal of the ML in this case is to identify patterns in the data, including:

 
 • 
 
 Clustering , one of the widely popular use cases, where the goal is to identify K K different clusters in the data. Since there is no unique metric to guide the clustering process, the choice of K K and final clusters may not represent the real world. The K-mean clustering is one of the simplest and most popular examples of clustering techniques; we discuss it below in the examples subsection.

 

 • 
 
 Dimensionality reduction . Usually, real-world data occur in high dimensional spaces, e.g., number of pixels in an image. This large dimensional space results in what is usually referred to as curse of dimensionality , where different phenomena start to show up in large dimensional spaces that are not important in low dimensional ones. An intuitive example is related to the number of examples needed to cover all the possible cases. For grayscale images, 2 d 2^{d} is the number of possible values for the d d features, which grows exponentially with the number of features [ 28 ] . Projecting the data onto lower-dimensional spaces is one method to reduce the size of the needed data. The Principle Component Analysis is one of the oldest and most effective techniques. We discuss it further in the examples section below.

 
 
 This problem can also be viewed as manifold learning problem. A manifold can be described, roughly, as a constraint shape in d d dimensions, but can be captured by k d k d dimensions. For instance, the points on the surface of a sphere, in a 3-Dimensional (3D) space, can be viewed as points on a bent 2D surface. In many practical problems, the observed data are on a manifold with an intrinsic dimensionality k k embedded in a d d dimensional space, see Fig. 2 . The dimension of the observed data can be reduced by unfolding the manifold in k k dimensional space. Learning the underlying manifold or some of its properties, such as its tangents, can be very helpful in understanding the relation and evolution of features. There is a number of algorithms to learn the manifold of the observed data; we provide examples in the following subsection.

 

 • 
 
 Density estimation. As discussed earlier, there is a number of advantages in knowing the underlying structure of the data, e.g., for constructing the graphical model, or regenerating data.

 

 • 
 
 Denoising. Here, the goal is to filter out the noise and reconstruct the original signal, i.e., the goal is to recover 𝒙 \boldsymbol{x} from a corrupted version of it 𝒙 ~ \tilde{\boldsymbol{x}} . Knowledge of the data (or noise) structure and density can thus help in this process.

 

 • 
 
 A number of the points above can be boiled down to the issue of data representation. The proper data representation depends on the problem. For instance, algorithms may handle different representations differently, e.g., some algorithms can utilize a sparse representation to improve performance, while others are oblivious to available sparsity and thus struggle with high dimensional data.

 

 
 We emphasize here that the above are not the only goals and are not mutually exclusive. For instance, clustering can be viewed as sparse representation of the data, where a point in a high dimensional features space can be represented by only one of K K clusters. Other examples are given below. Furthermore, although the dataset 𝒟 \mathcal{D} is assumed to have no labels, unsupervised learning can be viewed as prior step in a supervised learning problem, where the goal is to provide the right representation of the data.

 
 
 Examples: 

 
 • 
 
 K-means clustering is the most popular clustering algorithm. It assumes the number of clusters is known a priori, say K K , then it proceeds as follows. It randomly initializes the location of the clusters’ centroids 𝒄 k ∈ ℝ d \boldsymbol{c}_{k}\in\mathbb{R}^{d} , then assign the points 𝒙 i \boldsymbol{x}^{i} to the closest cluster based on some distance measure to the centroid. After all the data points have been assigned to clusters, it then updates the location of the centroid based on the data present in the current clusters, e.g., by giving it the value of the mean of the data in that cluster, i.e., 𝒄 k = ∑ i ∈ 𝒞 k 𝒙 i \boldsymbol{c}_{k}=\sum_{i\in\mathcal{C}_{k}}\boldsymbol{x}^{i} , where the set 𝒞 k \mathcal{C}_{k} includes the indices of data points in cluster k k . The procedure continues until no further changes of the cluster assignment occurs.

 

 • 
 
 Principal Component Analysis (PCA) is a linear mapping of the data of dimension d d to a subspace of dimension k k . The goal is to find a subspace of dimension k k that contains the largest variation of the data. In other words, PCA is a projection to the subspace (of dimension k k ) with the largest variation; by doing so it preserves most of the information but with a smaller dimensional space. When k = 1 k=1 , PCA finds the first ”principal component” 𝒖 ∈ ℝ d \boldsymbol{u}\in\mathbb{R}^{d} , such that var ⁡ ( 𝒖 ⊤ ​ 𝒙 ) {\rm var}(\boldsymbol{u}^{\top}\boldsymbol{x}) is the largest, where var ( . ) \rm var(.) refers to the variance over the distribution of 𝒙 \boldsymbol{x} . 𝒖 \boldsymbol{u} can be found to be the eigenvector that corresponds to the maximum eigenvalue of 𝒙 \boldsymbol{x} ’s d × d d\times d covariance matrix. In general, for k ≤ d k\leq d , and datasets in X = [ 𝒙 1 , … , 𝒙 n ] X=[\boldsymbol{x}^{1},...,\boldsymbol{x}^{n}] with dimension d × n d\times n , i.e., n n realizations of 𝒙 \boldsymbol{x} , the PCA can be found to be

 

 
 | 
 U = W ⊤ ​ X U=W^{\top}X | 
 | 
 

 where U U is the transformed data with size k × n k\times n , W W is the d × k d\times k projection matrix, with the columns of W W being the k k eigenvectors of the covariance matrix of 𝒙 \boldsymbol{x} that correspond to the k k largest eigenvalues.
Assuming 𝒙 \boldsymbol{x} values have mean zero, one can empirically approximate the covariance of 𝒙 \boldsymbol{x} by X ​ X ⊤ XX^{\top} , and then take the first k k eigenvectors of X ​ X ⊤ XX^{\top} that correspond to the largest k k eigenvalues of X ​ X ⊤ XX^{\top} , or equivalently take the first k k columns of left singular vectors of X X , found through Singular Value Decomposition (SVD).
Note there are other forms of PCA, such as kernel PCA, that can use a non-linear kernel to replace the inner product between data vectors, which allows to use the possible linear separability in higher dimension before mapping to lower dimension.

 

 • 
 
 Locally Linear Embedding (LLE) [ 29 ] is a nonlinear dimensionality reduction algorithm. It tries to discover the structure of the manifold using linear local relations between the data points. This is done by assuming that it is possible to reconstruct a point 𝒙 i \boldsymbol{x}^{i} by a convex linear combination of its neighbors 𝒙 j ∈ 𝒦 ⁡ ( i ) \boldsymbol{x}^{j}\in\mathcal{K}(i) . This relation should be preserved when it is mapped to a lower dimension, i.e., between the lower dimensional point 𝒄 i ∈ ℝ k \boldsymbol{c}^{i}\in\mathbb{R}^{k} and 𝒄 j ∈ 𝒦 ⁡ ( i ) \boldsymbol{c}^{j}\in\mathcal{K}(i) , where 𝒦 ⁡ ( i ) \mathcal{K}(i) is preserved between the two dimensions. Formally, for given neighbor relations between the points 𝒦 i \mathcal{K}_{i} , the solution can be found by two optimization problems:

 

 
 | 
 min ⁡ ∑ i w ⁡ | 𝒙 i − ∑ j ∈ 𝒦 i w i ​ j ​ 𝒙 j | 2 , s . t . ∑ j ∈ 𝒦 i w i ​ j = 1 ∀ i , \displaystyle\min_{w}\sum_{i}|\boldsymbol{x}^{i}-\sum_{j\in\mathcal{K}_{i}}w_{ij}\boldsymbol{x}^{j}|^{2},\penalty\ \penalty\ \penalty\ \penalty\ \penalty\ s.t.\sum_{j\in\mathcal{K}_{i}}w_{ij}=1\penalty\ \penalty\ \penalty\ \penalty\ \penalty\ \forall i, | 
 | 
 (6) | 
 

 which could be solved in closed form solution. Then for given weights w i ​ j w_{ij} , we need to solve for 𝒄 i \boldsymbol{c}^{i} :

 

 
 | 
 min ⁡ ∑ i 𝒄 ⁡ | 𝒄 i − ∑ j ∈ 𝒦 i w i ​ j ​ 𝒄 j | 2 \displaystyle\min_{\boldsymbol{c}}\sum_{i}|\boldsymbol{c}^{i}-\sum_{j\in\mathcal{K}_{i}}w_{ij}\boldsymbol{c}^{j}|^{2} | 
 | 
 (7) | 
 

 The 𝒦 i \mathcal{K}_{i} can be simply the K-nearest neighbors. Alternatively, 𝒦 i \mathcal{K}_{i} can be found using domain knowledge. An alternative representation of ( 7 ) is,

 

 
 | 
 min 𝑪 ⁡ | 𝐂 ⁡ ( 𝐈 N − 𝛀 ) | 2 \min_{\boldsymbol{C}}|{{\bf C}({\bf I}_{N}-{\bf\Omega})}|^{2} | 
 | 
 

 where 𝐂 = [ 𝒄 1 , … , 𝒄 N ] {\bf C}=[\boldsymbol{c}^{1},...,\boldsymbol{c}^{N}] , is a k × N k\times N matrix, 𝐈 N {\bf I}_{N} is an identity matrix of size N × N N\times N , and 𝛀 = [ 𝝎 ~ 1 , … , 𝝎 ~ N ] {\bf\Omega}=[\tilde{\boldsymbol{\omega}}^{1},...,\tilde{\boldsymbol{\omega}}^{N}] is a N × N N\times N matrix, such that a vector 𝝎 ~ i \tilde{\boldsymbol{\omega}}^{i} has the j th j^{\rm th} element equal to ω i ​ j \omega_{ij} and the rest are zero. We can define L = 𝐈 N − 𝛀 L={\bf I}_{N}-{\bf\Omega} as the graph Laplacian . This is an important matrix that is the central component in Laplacian eigenmap dimensionality reduction algorithms that are used widely in localization, where the goal is to directly optimize

 

 
 | 
 min ⁡ ∑ i , j 𝒄 ⁡ ω i ​ j ​ ( 𝒄 i − 𝒄 j ) 2 ≡ min 𝐂 ⁡ 𝐂 ⊤ ​ 𝐋𝐂 \displaystyle\min_{\boldsymbol{c}}\sum_{i,j}\omega_{ij}(\boldsymbol{c}^{i}-\boldsymbol{c}^{j})^{2}\equiv\min_{\bf C}{\bf C^{\top}LC} | 
 | 
 (8) | 
 

 Here the weights can be set (or identified) from the constructed weighted graph that capture the correlation of the vertices (i.e., 𝒙 i \boldsymbol{x}^{i} ’s).

 

 
 
 
 Figure 2: A visualization for high dimensional wireless data, revealing a possible structure of the underlying manifold. Data generated using ray-tracing software at 2.4 2.4 GHz. (a) The transmitter (Basestation) and receiving points locations, with point color coded to differentiate the different streets. (b) Dimensionality reduction using t-SNE (t-Distributed Stochastic Neighbor Embedding) [ 30 ] . For each point we used 30 different features including angles of arrival, delay and power of the 5 best MPCs (see sec. II-B ). Notice there is a reasonable agreement between the locations (streets) and the points after reducing the dimension, this is more prominent for the streets close the basestation. More about the setups can be found in [ 31 ] . 
 
 
 

#### II-A 3 Other Learning Approaches

 
 As highlighted above, supervised and unsupervised learning are not the only classes of machine learning approaches. Other include

 
 • 
 
 Semi-supervised learning . In such approach the task is usually similar to supervised learning, however, the dataset includes both labeled and unlabeled data, i.e., the dataset 𝒟 = { 𝒙 i , y i } i = 1 N ∪ { 𝒙 j } j = 1 M \mathcal{D}=\{\boldsymbol{x}^{i},y^{i}\}_{i=1}^{N}\cup\{\boldsymbol{x}^{j}\}_{j=1}^{M} , has N N and M M labeled and unlabeled data points, respectively.

 

 • 
 
 Reinforcement learning . The system observes a stream of data with partially observable labels or sometimes only feedback values that indicate the quality of the system’s decision (actions).

 

 • 
 
 Online Learning . The data is only available during system operation.

 

 • 
 
 Transfer learning (TL). The available data include a large number of data examples from a source domain that is ”close” to the domain of interest ( target domain), and a few examples from the target domain. By domain we refer here to the observable features and the task of the ML problem. In TL problems at least characteristics of the data (e.g., distribution p ⁡ ( 𝒙 d ) p(\boldsymbol{x}_{d}) or p ⁡ ( y d ) p(y_{d}) ), the mapping (e.g., p ⁡ ( y d | 𝒙 d ) p(y_{d}|\boldsymbol{x}_{d}) ) could be different between the two domains, we used the subscript d d to refer to the domain specific feature and labels.

 

 
 
 
 

#### II-A 4 Deep Learning

 
 There is no unique definition of DL but it usually refers to ML techniques that use hierarchical learning structures. DL has recently shown remarkable performance in a number of challenging fields, such as computer vision (CV) and natural language processing (NLP). Being a subset of ML techniques, DL has been incorporated in number of the different classes of ML, such as supervised, unsupervised etc. Many of the recent successful architectures are based on Neural Networks (NNs), thus in the following we describe NNs, their basic architectures, and how they are trained. Next, we summarize a number of popular architectures that have been used in localization problems. Each is a representative of certain interesting properties in DL. We then summarize novel solutions to a number of challenges in DL.

 
 
 Figure 3: Examples of different deep NN architectures. 
 
 
 Figure 4: Three different activation functions (e.g., referred to in Fig. 3 as σ \sigma functions in various DL architectures). 
 
 
 Feedforward Neural Networks 
Artificial NNs are powerful structures designed to imitate the human brain. An NN consists of one or more layers, each of which has a number of parallel neurons (nodes), see Fig. 3 -(a). The neuron computes a weighted combination of the input features and then passes it through a (usually non-linear) transformation, a.k.a. ”activation function”. Fig. 4 shows examples of such nonlinear activation functions. Thus a neuron can be viewed as a generalization of logistic regression (see above), or a perceptron (uses threshold based activation function).

 
 
 When NNs have no feedback loops they are referred to as Feedforward NN. The simplest architectures are sometimes referred to as fully-connected multi-layer perceptrons (MLP), where each node (perceptron) in a layer is connected to all other nodes in the following layer, see Fig. 3 -(a). One reason for their popularity can be attributed to the universal approximation theorem, which states that with at least one hidden layer and non-linear activation functions (such as sigmoid), NNs can approximate any function with arbitrary accuracy [ 32 ] . However, to achieve such results with single hidden layer, it could be necessary to have an very large number of neurons [ 24 ] . Increasing the depth of the network can alleviate this problem, as the number of possible network states grow exponentially with the depth of the network [ 24 ] .

 
 
 Even to train a single neuron, similar to the logistic regression example, we need an optimization algorithm. For the training of multiple neurons and deep architectures, efficiency of such training is critically important. NN training commonly uses backpropagation, which consists of forward pass of the training input values 𝒙 i , i ∈ { 1 , . . , n } \boldsymbol{x}^{i},i\in\{1,..,n\} . Then, utilizing the chain rule, we back-propagate the gradient of the loss function over the layers; subsequently algorithms such as gradient descent can be used to update the parameters. Since many DL problems use large datasets, the parameters update is usually done over mini-batches (compared to using all the data at once to calculate the gradient in each iteration as a single batch); this is usually referred to as Stochastic Gradient Descent (SGD). There are various SGD based optimizers; methods can use different functions of the gradients, e.g., the weighted average of the gradients. The Adam (Adaptive Moment Estimation) optimizer is one of the most widely used algorithms [ 24 ] .

 
 
 Convolutional NN (CNNs): are efficient Feedforward NNs architectures, see Fig. 3 -(d), that have shown outstanding performance in CV [ 33 ] . Compared to a fully connected NN, CNNs use parameter sharing, which allows building deeper networks with a much smaller number of parameters. Key components in CNNs are the ”filters” (sometimes referred to as kernels or features map) and the convolution operation. The filters have tunable weights that are multiplied with the input data during the convolution process. However, different from the fully connected NN, the size of the filter could be much smaller than the size of the input data. To demonstrate that, let us consider an example from CV: For an image of size 1024 × 768 × 3 1024\times 768\times 3 (or generally n × m × k n\times m\times k ), corresponding to height, width and color dimensions of an image, 5 5 
 5 
 
 
 
 Note here we deviated from our initial definition of an example 𝒙 ∈ ℝ d \boldsymbol{x}\in\mathbb{R}^{d} to 𝒙 ∈ ℝ n × m × k \boldsymbol{x}\in\mathbb{R}^{n\times m\times k} , as each data-point (i.e., example) represents a tensor. we could apply a filter of size 2 × 3 × 3 2\times 3\times 3 (or generally a × b × c a\times b\times c ), where a n , b m a n,b m . Note that the third dimension is sometimes referred to as ”channels”. The filter is then ”convolved” with the input data, where the filter can start from, for example, the left upper corner of the image, then the filter’s weights and the input data values in the over lapping region are correlated (element-wise multiplied followed by summation), the filter is then shifted by s s pixels to the right and then correlated with the overlapping pixels. The value of s s is usually referred to as the stride . The process continues to produce an output of size ⌊ n − a s + 1 ⌋ × ⌊ m − b s + 1 ⌋ × 1 \lfloor\frac{n-a}{s}+1\rfloor\times\lfloor\frac{m-b}{s}+1\rfloor\times 1 , the last dimension (number of channels) is equal to the number of filters used at this layer.

 
 
 We should here point out two interesting observations from the example above: (1) the parameter sharing, where all the pixels share the same set of weights. (2) Connection sparsity, where each output value is produced by a × b × c a\times b\times c elements. These properties contribute to interesting features of the CNNs: translation invariance and efficiency. The translation invariance refers to the robustness of the CNNs to translation, and thus produces similar responses regardless of the shift of the input data. The efficacy allows to build deeper architectures, and thus reap the advantages of the network depth as discussed above. Also the reduction of the number of trainable parameter reduces the chances of over-fitting, making the network trainable with a smaller number of data examples. For example, AlexNet , which is one of the most popular image classification models, is built with 5 convolutional layers with about 60 million parameters; it has achieved remarkable performance in classifying images in ImageNet dataset into 1000 classes. We finally note that many problems in localization can be re-framed as image classification, e.g., RSSI values are placed into a 2D grid and the output is the corresponding fingerprint; as will also be discussed in later sections.

 
 
 Recurrent NN 
Recurrent Neural Networks (RNNs) are a type of NN widely used in modelling sequential data. They have shown great promise in time series data processing and NLP. RNNs capable of performing tasks in which the current output is dependent on both the previous outputs and the current input. RNNs can be thought of having the ability to “remember” what has been calculated so far.
RNNs have a hidden state 𝒉 ⁡ ( t ) \boldsymbol{h}(t) , which can store information about the past computations. The hidden state is a function of the previous hidden state and the current input and is calculated by

 

 
 | 
 𝒉 ⁡ ( t ) = σ h ​ ( V x ​ 𝒙 ​ ( t ) + V h ​ 𝒉 ​ ( t − 1 ) + 𝒃 h ) , \boldsymbol{h}(t)=\sigma_{h}(V_{x}\boldsymbol{x}(t)+V_{h}\boldsymbol{h}(t-1)+\boldsymbol{b}_{h}), | 
 | 
 

 where σ h ​ ( ) \sigma_{h}() is the activation function, V h V_{h} and V x V_{x} and b h b_{h} are trainable parameters. The output at time t t , y ⁡ ( t ) y(t) , can then be calculated using 𝒉 ⁡ ( t ) \boldsymbol{h}(t) as:

 

 
 | 
 y ⁡ ( t ) = σ y ​ ( W h ​ 𝒉 ​ ( t ) + 𝒃 y ) , y(t)=\sigma_{y}(W_{h}\boldsymbol{h}(t)+\boldsymbol{b}_{y}), | 
 | 
 

 where σ y \sigma_{y} is the activation function.
The weights in RNNs are shared across time. For example, V h V_{h} and W h W_{h} ( 𝒃 h \boldsymbol{b}_{h} and 𝒃 y \boldsymbol{b}_{y} ) are shared among all connections in a layer. At each point in time, the back propagation through time (BPTT) algorithm is used for training and adjusting the weights in each iteration.
However, RNNs usually suffer from gradient vanishing or explosion, where the gradient values decay (or grow) as we propagate the gradient through time. An even bigger problem is the decaying influence over time i.e., the model cannot remember too far in the past. Hence vanilla RNNs are not widely used in practical applications.
In practice, decaying influence or degradation over time is addressed by using “gates” in the RNN cells. Long Short Term Memory (LSTMs) and Gated Recurrent Units (GRUs) are the two most popular gated RNN models, see Fig. 3 -(c) for an example of an LSTM architecture, where the network uses three gates to control what to remember in the memory cell, when to use it, and when to forget it. There are many variations and designs of RNNs, such as bi-directional RNNs. Examples of successful deep architectures based on RNNs are sequence-to-sequence networks [ 34 ] , Conv-LSTM [ 35 ] . In localization, RNNs are of obvious importance for tracking the location of targets over time.

 
 
 Auto-Encoders 
Auto-encoders (AEs) are NN architectures where the output of the of the network is a ”copy” of the input data, see Fig. 3 -(b). Thus, AEs are usually considered as unsupervised learning architectures. We can distinguish three main components of the AEs, (i) the encoder g ( . ) g(.) , (ii) the latent variable 𝒉 \boldsymbol{h} , and (iii) the decoder f ( . ) f(.) . The encoder maps the input 𝒙 \boldsymbol{x} data to the hidden state 𝒉 \boldsymbol{h} , the decoder maps 𝒉 \boldsymbol{h} back to 𝒙 \boldsymbol{x} . In practice, we are interested in 𝒙 ~ ~ = f ⁡ ( g ⁡ ( 𝒙 ~ ) ) \tilde{\tilde{\boldsymbol{x}}}=f(g(\tilde{\boldsymbol{x}})) , where the relations between 𝒙 \boldsymbol{x} , 𝒙 ~ \tilde{\boldsymbol{x}} and 𝒙 ~ ~ \tilde{\tilde{\boldsymbol{x}}} depend on the application at hand. For denoising AEs, we want the AE to denoise the input signal, such that 𝒙 ~ ~ ≈ 𝒙 \tilde{\tilde{\boldsymbol{x}}}\approx\boldsymbol{x} given that the data used at the input is noisy version of 𝒙 \boldsymbol{x} , i.e., 𝒙 ~ = 𝒙 + 𝒏 \tilde{\boldsymbol{x}}=\boldsymbol{x}+\boldsymbol{n} , where 𝒏 \boldsymbol{n} represents the noise vector. In this case, the loss function defined in eq. ( 5 ) has y = 𝒙 y=\boldsymbol{x} , i.e., the ”label” here is the noise-free data. The training procedure follows the standard backpropagation algorithm that we described earlier.

 
 
 In another important application, we might be interested in the latent representation 𝒉 \boldsymbol{h} , which is sometimes referred to as ”code”. This can be seen as dimensionality reduction when the size of 𝒉 \boldsymbol{h} is much smaller than that of 𝒙 \boldsymbol{x} . In this case our goal is to train the AE such that 𝒙 ~ ~ ≈ 𝒙 \tilde{\tilde{\boldsymbol{x}}}\approx\boldsymbol{x} , where here 𝒙 ~ = 𝒙 \tilde{\boldsymbol{x}}=\boldsymbol{x} . For these applications, we hope that 𝒉 \boldsymbol{h} provides a robust concise representation or a projection on the underlying manifold. Note that when the neurons of the AEs use linear activation functions, AEs can be shown to be equivalent to PCA. With nonlinear activation functions, AEs can provide powerful non-linear dimensionality reduction.

 
 
 There are several other kinds and architectures of AEs, e.g., penalizing the hidden representation 𝒉 \boldsymbol{h} leads to a sparse AE, i.e., by adding an L 1 L_{1} regularization to the cost function r ⁡ ( 𝒉 ) = | 𝒉 | 1 r(\boldsymbol{h})=|\boldsymbol{h}|_{1} , compare to ( 5 ). This can be used to learn important features and be used as method for pre-training other NN based architectures [ 24 ] . Another example is a generative AE network named variational AE (VAE), which is derived based on the variational inference for generative models. VAEs take samples from a random distribution p ⁡ ( 𝒉 | 𝒙 ) p(\boldsymbol{h}|\boldsymbol{x}) (i.e., the encoder) to generate examples of 𝒙 \boldsymbol{x} . The VAE utilizes what is known to be the ”reparameterization trick”, that allows to separate the stochastic sampling process from the parameters of q ⁡ ( 𝒉 | 𝒙 ; Φ ) q(\boldsymbol{h}|\boldsymbol{x};\Phi) , so that we backpropagate through the parameters and train the model as we do for Feedforward NNs.

 
 
 Restricted Boltzman Machine and Greedy Pretraining 
Restricted Boltzman machines (RBM) are simple undirected graphical models that are used to learn the distribution of the data p ⁡ ( 𝒙 ) p(\boldsymbol{x}) . They consist of one layer of observable data and one layer of hidden data, with no connections within the same layer. The probability relation between the vertices are captured with an energy based model

 

 
 | 
 p ⁡ ( 𝒙 , 𝒛 ) = 1 Z ​ exp − E ⁡ ( 𝒙 , 𝒛 ) p(\boldsymbol{x},\boldsymbol{z})=\frac{1}{Z}\exp^{-E(\boldsymbol{x},\boldsymbol{z})} | 
 | 
 

 with E ⁡ ( 𝒙 , 𝒛 ) E(\boldsymbol{x},\boldsymbol{z}) represent the ”energy” between the variables which have trainable parameters, and Z Z is a, usually intractable, partitioning function that ensures that p ⁡ ( 𝒙 , 𝒛 ) p(\boldsymbol{x},\boldsymbol{z}) is a proper probability. The bipartite nature of the graphs makes RBM relatively easy to be trained by approximating the gradients of Maximum Likelihood using ”Contrastive Divergence” (iterative procedure uses Gibbs sampling to sample the model) [ 24 ] .

 
 
 Greedy layer-wise pre-training was one of the early methods to allow training deep networks. The network is built layer-wise, where initially, the first hidden layer represents the hidden layer RBM (or the hidden layer is of that for one layer AE). Once trained, the hidden layer will be the input layer of a new hidden layer that will be added on the top of the current network. This process continues until the network structure is constructed. This provides good initial weights and acts as regularization [ 24 ] . However, the modern training and regularization methods have over-shadowed these techniques.

 
 
 Techniques to Enhance DL Solutions 
The hierarchical and deep structure of DL solutions exacerbates some of the existing ML challenges and brings many new ones. For instance, in DL, training networks with tens of million of parameters is not unheard of (AlexNet has over 60 million parameters), thus aggravating the over-fitting problem. In the last decade many novel techniques were proposed to alleviate some of these challenges. Here we briefly list some of them:

 
 • 
 
 Dropout is one of the novel techniques to combat over-fitting, where some of the neurons are shut off at random during training. This increases the network robustness as it is supposed to provide a solution even if some of its internal connections are not available, or highly noisy (think of product noise). One interesting view of dropout is that it can be viewed as bagging with exponentially many NNs [ 24 ] .

 

 • 
 
 Noise injection: here we can corrupt the data with several realizations of noise. While artificial addition of noise seems counter-intuitive - in particular as many systems try to reduce the noise during preprocessing step earlier - it
increases the number of available data and increases the network’s robustness to small variations.

 

 • 
 
 Data augmentation. Since overfitting can occur due to the limited number of available examples, one solution is to increase the size of the dataset. However, collecting new data is usually costly. In data augmentation we aim at generating new data examples from the existing ones. Noise injection above is one such technique. Another example is shifting, rotating or cropping images in CV. We here emphasize that data augmentation techniques that are valid in one domain may not be applicable for other problems.

 

 • 
 
 Early stoppage. One intuitive and efficient solution to overfitting is to stop the iterative training algorithm before it starts fitting the parameters over the noise and the outliers. Finding the right stopping time can be assessed by evaluating the performance over a validation set.

 

 • 
 
 Utilizing related data through pre-training. Typically data from a different domain, or unlabeled data could be available in abundance. Pre-training over such data could be very helpful. Briefly, some of these benefits include:

 
 – 
 
 Provide good weight initialization, since during the training process we run an iterative optimization.

 

 – 
 
 Utilize low level features that could be shared over different domains. This is also related to the concept of Transfer Learning.

 

 – 
 
 Allow the network to estimate the underlying density of the data and possible correlation. This is usually viewed to be in the domain of unsupervised learning.

 

 
 

 
 
 
 

#### II-A 5 Available DL Platforms

 
 One reason behind the accelerated research and development of DL based solutions can be attributed to accessibility of suitable open-source platforms and libraries in common programming languages, some of which were initially developed by the industry. For DL, Python is the most widely used programming language, as per a GitHub survey.
This can be attributed to the fact that Python syntax is easy to learn and apply,
so that more tools and frameworks for DL are available in Python compared to other languages.
 C++ is the second-most popular language for ML, and in particular applied
where efficiency is key and speed is needed.
 C++ can fully exploit the power of GPUs at the very low level of programming. Python does not provide such flexibility. Java and other languages, including R and MATLAB , are also used at times.

 
 
 The most widely used libraries are Numpy , SciPy and Pandas ; all of which are Python libraries. These are normally used for intense vector math and are very efficient.

 
 
 TensorFlow and PyTorch are two of the most widely used DL libraries. They support a wide array of operations and provide high flexibility for doing almost anything in DL. Another famous framework is Keras , which can be used with a backend of TensorFlow or Theano . Below are some comparisons of these three: 

 
 
 
 • 
 
 While TensorFlow and PyTorch are DL libraries, Keras is a framework made just for DL. This means TensorFlow and PyTorch can be used for vector math, but Keras is made to make DL easier.

 

 • 
 
 TensorFlow works on a backend of Theano , whereas PyTorch works on a backend of Torch , both of which can be used as DL libraries themselves. Keras can be used with a backend of Theano as well as TensorFlow . 

 

 • 
 
 Prior to TensorFlow 2.0 , TensorFlow did not support eager execution. This is because TensorFlow used static computational graphs, which cannot be changed once created. So execution of operations would require a tf.Session() or an Interactive Session. PyTorch and Keras both don’t need such a Session (because PyTorch graphs are dynamic).

 

 • 
 
 PyTorch is generally used in research projects and TensorFlow is used in production systems. 

 

 • 
 
 The three frameworks are all open-source.

 

 
 Note that we focused here on the three frameworks above that generally have the largest number of users along with a very strong online community support. However, there are other frameworks such as Caffe , MxNet and Theano .

 
 
 
 

### II-B Wireless Channels 

 
 RF-based localization systems rely on the properties of the wireless propagation channel such as the attenuation or the flight time of a signal traveling from the transmitter (TX) to the receiver (RXs).
In free space, where a single electromagnetic wave carries the signal without interaction with other objects, these properties can be easily related to the location of TX and RX.
However, real-life propagation channels are much more complicated, a fact that creates the main challenges for precision localization. In this subsection we thus review the most important propagation characteristics that have an impact on localization systems. Furthermore, since many localization systems are integrated into communications systems, we also present a brief overview of the most salient features of those systems.

 
 
 A wireless RF signal emitted from the TX antenna(s) interacts with the environment through reflection, scattering, and diffraction at various objects in the environment before it arrives at the RX side. These processes give rise to, and determine the properties (such as amplitude, phase, delay, and angle) of, the multi-path components (MPCs) [ 36 ] . The different MPCs, which are usually modeled as plane waves, have different amplitude | α | |\alpha| , phase shift ϕ \phi , delay τ \tau , angle-of-departure (AoD) Ω \Omega from the TX, and angle-of-arrival (AoA) Ψ \Psi at the RX. The propagation channel , more precisely, the double-directional channel impulse response, can thus be written as a sum of N ⁡ ( t ) N(t) plane waves (MPCs) [ 37 ] 

 

 
 | 
 h ⁡ ( t , τ , Ω , Ψ ) = ∑ l = 1 N ⁡ ( t ) α l ​ δ ​ ( τ − τ l ) ​ δ ​ ( Ω − Ω l ) ​ δ ​ ( Ψ − Ψ l ) , h(t,\tau,\Omega,\Psi)=\sum_{l=1}^{N(t)}\alpha_{l}\delta(\tau-\tau_{l})\delta(\Omega-\Omega_{l})\delta(\Psi-\Psi_{l}), | 
 | 
 (9) | 
 

 where | α | , τ , Ω , Ψ |\alpha|,\tau,\Omega,\Psi are constant within a small area (typically a few meter) called ”stationarity region”. When a TX or RX moves over larger distances, also the | α | |\alpha| (due to shadowing and changing pathloss), and the τ , Ω , Ψ \tau,\Omega,\Psi (due to changes in the geometry) change. Furthermore, when the considered frequency changes by a large amount (typically 10 % 10\% of the carrier frequency) called ”stationarity bandwidth”, the | α | |\alpha| become a function of frequency, or - equivalently - the δ ⁡ ( τ − τ l ) \delta(\tau-\tau_{l}) is replaced by a function describing the delay dispersion of a single MPC ξ ⁡ ( τ − τ l ) \xi(\tau-\tau_{l}) ; this situation is relevant for ultrawideband (UWB) systems [ 38 ] . Note that any impulse response can be expanded into a sum of plane waves as given above, yet in some cases it is more informative (and sometimes better related to the underlying physics) to describe a large number of weak components by a continuous ”diffuse multipath component” DMC [ 39 ] . Finally, we note that a further generalization can be achieved by incorporating the polarization of the MPCs, which converts ( 9 ) into a matrix equation [ 40 ] . We also note that a Fourier Transform (FT) with respect to τ \tau provides the channel frequency response (CFR) h ⁡ ( t , τ , Ω , Ψ ) → ℱ τ H ⁡ ( t , f , Ω , Ψ ) h(t,\tau,\Omega,\Psi)\xrightarrow{\mathcal{F_{\tau}}}H(t,f,\Omega,\Psi) .
A FT with respect to t t provides a transformation to the Doppler shift ν \nu , resulting in the Doppler-variant impulse response, also known as ”spreading function” h ⁡ ( t , τ , Ω , Ψ ) → ℱ t H ⁡ ( ν , τ , Ω , Ψ ) h(t,\tau,\Omega,\Psi)\xrightarrow{\mathcal{F}_{t}}H(\nu,\tau,\Omega,\Psi) .

 
 

#### II-B 1 Impact of the antenna and system

 
 The above description describes the propagation channel only, without incorporating the effect of the antennas or the RF components. The signals that are actually observed at the antenna connectors of an array are related to the double-directional impulse response as

 

 
 | 
 h m , n ​ ( t , τ ) = ∫ d ​ Ψ ​ G RX , n ​ ( Ψ ) ​ ∫ d ​ Ω ​ G TX , m ​ ( Ω ) ​ h ​ ( t , τ , Ω , Ψ ) h_{m,n}(t,\tau)=\int d\Psi G_{{\rm RX},n}(\Psi)\int d\Omega G_{{\rm TX},m}(\Omega)h(t,\tau,\Omega,\Psi) | 
 | 
 (10) | 
 

 where G TX , m ​ ( Ω ) G_{{\rm TX},m}(\Omega) and G RX , n ​ ( Ψ ) G_{{\rm RX},n}(\Psi) are the complex antenna pattern of the m-th TX and n-th RX antenna element, respectively. For the case of single-antenna elements, we simply set m = n = 1 m=n=1 .
We can see from this that the measured impulse response is not only a function of the propagation channel, but also of the antennas, and that measurements with different antennas thus result in different impulse responses even for the same propagation channel, i.e., the same location.

 
 
 Furthermore, the necessarily finite bandwidth of the transmit waveform and receive filter leads to a superposition of the MPCs that arrive at approximately (within one inverse bandwidth) at the same time; this superposition can be constructive or destructive depending on the different phase shifts. Thus, the actually measured ”channel impulse response” (CIR) impulse response is

 

 
 | 
 h meas , m . n ​ ( t , τ ) = h sys ​ ( τ ) ∗ h m , n ​ ( t , τ ) . h_{{\rm meas},m.n}(t,\tau)=h_{\rm sys}(\tau)\ast h_{m,n}(t,\tau)\penalty\ . | 
 | 
 (11) | 
 

 where " ∗ " "*" is the convolution operator, and h sys h_{\rm sys} is the effective system impulse response. Obviously, the smaller the bandwidth, the more MPCs are superimposed by this convolution operation, and the temporal resolution (ability to find the delay of the MPCs decreases).
A similar observation can be made for directional channel characteristics: a suitable FT can move the observations from different antenna elements (location space) into the ”beamspace”, providing the same results as if we would have a set of co-located directional antennas that are pointing into different directions [ 41 ] . Each of those virtual directional antennas superposes the MPCs that are in its beamwidth. Thus, the smaller the antenna aperture (and thus, the larger the beamwidth), the more MPCs are superposed, and the worse the directional resolution.

 
 
 

#### II-B 2 Small Scale vs Large Scale Fading 

 
 The observables, h meas , m . n ​ ( t , τ ) h_{{\rm meas},m.n}(t,\tau) , experience what is typically called small scale and large scale fading. Small scale fading is caused by the fast variations of the phase (due to movement of the TX, RX and/or objects in the environment) and resulting change in constructive/destructive summation. The amplitude (power) variations of the signal due to small scale fading can be 30 dB or more.
A movement of the TX or RX on the order of a wavelength is usually sufficient to change a constructive to a destructive interference or vice versa. This may be advantageous, e.g., for a fingerprinting-based localization system because it provides a very sensitive measure of location. Yet, small-scale fading also has major drawbacks. For example,
even when TX and RX are completely static, the movement of interacting objects (e.g., moving pedestrians, cars, etc.) can significantly change the small scale fading state. Thus, for a fingerprinting system, the observables are sensitive to small environmental changes beyond the control of the system. In addition to fading, multipath propagation also leads to delay dispersion or, equivalently, frequency selectivity. The former means that even when we send a a very short pulse from the TX, the received signal extends over a considerable delay range; the latter means that at different frequencies, we observe different values of the small-scale fading. Multipath propagation has been generally considered an obstacle in deterministic localization procedures, though recent work has established ways of turning it into an advantage, see [ 42 ] and references therein.

 
 
 Besides the small scale fading, there are also larger-scale variations that are caused by the fact that the other MPC parameters (power, angle, delay) change as the TX and/or the RX move over larger distances. The large-scale characteristics, such as the angular power spectrum (power as a function of the incident directions, averaged over the small scale fading), are thus indicative of the general area in which a device is located. On an even larger scale occurs the pathloss, which is a power loss due to the ”thinning out” of the area power density as waves propagate further and further away from the TX. In the case of a pure ”free-space” scenario, this power loss is proportional to the square of the TX-RX distance d d ; more generally it can be modeled as d n PL d^{n_{\rm PL}} , where the pathloss coefficient n PL n_{\rm PL} depends on the environment and whether a LOS connection exists between TX and RX.

 
 
 

#### II-B 3 Channel Models

 
 There are a number of models for wireless propagation channels that can be used to test localization algorithms. Most important among these are ”statistical channel models”, which prescribe the small-scale and large-scale statistics of the propagation channel in closed form (or tabular form), based on which different realizations of the channel can be created. The most widely used of those models are the 3GPP channel models, originally designed for 3G cellular communications systems [ 43 , 44 ] , and repeatedly extended over the years, so that they are now used for 4G and 5G cellular systems as well. An efficient and widely adopted implementation is provided by the Quadriga website quadriga-channel-model.de, see also [ 45 ] . The model generates double-directional channels for a ”drop” of a UE (user equipment, mobile station), i.e., placing of a UE in a particular location within a cell. The ”baseline” direction is the LOS between the base station (BS) and the UE, and all angles are defined relative to this direction. The double-directional impulse response consists of a number of paths , each of which has a particular delay, which is chosen at random, according to a given (parameterized) probability density function. Power is assigned according to the path delay, with the average power decreasing with increasing delay. Furthermore, each paths consists of 20 sub–paths, which all have the same delay, but slightly different angles, such that each path has an angular spread of, e.g., 5 degrees. Each of the sub–paths has the same amplitude, and random phases; their superposition thus provides not only an angular spread, but also small–scale fading when either different values of the random phases are chosen or the UE moves. In the latter case, the geometric relationship between the antenna array orientation, direction of the sub-paths at the UE, and the UE movement vector allow to compute the temporal changes of the impulse response.

 
 
 While, of course, the propagation channel is independent of which applications are run over it, the admissible simplifications of a channel model do depend on the application.
Thus, it must be kept in mind that the 3GPP model was designed to test communications systems, and not localization systems. For example, the average power in the first path is assumed to be the largest, even though extensive measurements have shown situations with a ”soft onset” (e.g., [ 46 ] ), and other situations where there is no discernible energy at the LOS arrival time, and the first component is only several ns later [ 47 ] . Furthermore, changes of the channel when the UE moves over large distances are not modeled accurately (the recently introduced ”spatial consistency” simulation approach solves the problem only partially), and the purely stochastic modeling of shadowing based on a simple (exponential correlation) shadowing model also fails to model interesting higher-order relations between channels at widely separated locations. Last but not least, the channel model has a finite number of MPCs that are all fairly strong, and no DMC. As a consequence of all these simplifications, the channels generated by the 3GPP model might not show the same complexity as real-world (measured) channels, which might have significant impact on localization in general, and ML-based localization in particular.

 
 
 A channel model that is somewhat more realistic for localization systems is the IEEE 802.15.4a model [ 48 ] , which was designed explicitly for the testing of both communications and localization. It is also a statistical channel model, but considers the ”soft onset” of MPCs. However, it does not include directional characteristics, and is furthermore not intended for outdoor usage.

 
 
 
 
 

## III Types of Localization 

 
 The ML-based localization solutions that have been proposed in the literature differ depending on the considered localization level, as well as the target engagement. In the following subsections, we discuss those two aspects; Table I lists papers based on these categorizations. We defer the discussion of the conventional localization systems vs. the ML-based ones to Sec. IV-F , i.e., after we introduce the basic RF features in next section.

 
 

### III-A Localization Level 

 
 We here differentiate between three types: (i) Region classification, (ii) Fingerprint classification, (iii) and coordinates estimation.
Region classification means that the system aims to identify a sub-region of the study area in which the target is located, such as the building, floor, or room. This can be interpreted as finding the location with a certain, more or less rough, quantization, or, in other words, a classification problem. This result can be either part of a multi-step localization, i.e., forming the starting point for a more accurate coordinate-level localization, or used by itself. As a matter of fact, in many situations this result will be sufficient. For example, in buildings with many small rooms (e.g., hotels) identifying the target at room level will be sufficient.
The main advantages of this approach are (a) low misclassification error, and (b) possibly reduced computational effort.

 
 
 Alternatively, in fingerprint localization, the solution could be to match the target to the closest fingerprint in the database, also known as Reference Point (RP). Also this solution approach can be viewed as classification problem with N N distinguished classes, where N N is the number of RPs in the database. The localization error (in meter) that is engendered by such an approach depends on many factors such as the separation distance between RPs in the database, the environment and the matching method. Note especially that a small error in the fingerprint could lead to a large error in the localization.
To provide a better accuracy, several work have considered the fusion of the closest K K locations of target locations. This reduces the probability of associating with a far-away fingerprint, and furthermore reduces the quantization error, since a location interpolation between the set of nearby RPs can be performed.

 
 
 The most natural localization solution is to provide the explicit coordinates of the target; in the context of ML this can be seen as a regression problem. The approach can either use as input the signals at/from all the anchors, and deduce the target location directly, or it can learn the relation between the input features and the distance in considered dimensions, e.g., the mapping between the observed time of signal arrival and the distance, which then can be fed to a classical trilateration algorithms.

 
 
 There have been several other solutions that use ML to aide the localization process, such as LOS vs Non-LOS (NLOS) discrimination [ 49 , 50 ] , or scheduling of localization signals [ 51 ] ; however, these applications are not at the core of this survey.

 
 
 

### III-B Target Engagement 

 
 Depending on the application, localization systems could differ in the level of target engagement in the process. We here classify the localization to be

 
 • 
 
 Active localization : here the target is equipped with a wireless transceiver, and participates in the localization process. Due to the operating principle, different targets can be easily differentiated. While normally only anchors and targets exchange information, improved accuracy can be achieved when multiple targets cooperate with each other ( cooperative localization).
For future reference, we furthermore define

 
 – 
 
 One-way localization: in this approach, the target either sends out a localization (training signal) that is received by the anchor node(s) of the system, or conversely it receives localization signals from the anchor nodes.

 

 – 
 
 Two-way localization: in this approach, the target both transmits to and receives from the anchors signals that can serve to localize.

 

 
 

 • 
 
 Passive localization : here, the target does not have a wireless transceiver, but rather only reflects a signal; this is sometimes referred to as device-free localization, and more often as radar. Different targets are not necessarily easy to distinguish from their signal characteristics - for example, cars of similar type have similar radar signatures. 6 6 
 6 
 
 
 
 ML for radar is outside the scope of the current paper, as it focuses on different aspects such as detection and identification of objects with possible presence of jamming systems, some of which are related to remote sensing and image processing. A comprehensive survey is provided, e.g., in [ 4 ] . The coupling of the two could be interesting research direction. 

 

 • 
 
 Semi-passive localization : a borderline case are passive RFID tags. While they are passive in the sense that they do not contain transceivers and only reflect signals, they do so in a way that is unique for a particular tag, and thus allow distinction of the signals from different targets.

 

 
 Table I shows a list of relevant papers.

 
 
 
 
 
 | 
 
 
 Classification Region 
 | 
 
 
 Classification RP 
 | 
 
 
 Coordinates 
 | 

 
 
 
 
 
 Passive 
 | 
 
 
 [ 52 , 53 ] 
 | 
 
 
 [ 54 , 55 , 56 , 57 , 58 , 59 , 60 , 61 , 62 ] 
 | 
 
 
 [ 63 , 53 , 64 , 65 ] 
 | 

 
 
 
 Active 
 | 
 
 
 [ 66 , 67 , 68 , 69 , 70 , 71 , 72 , 73 , 74 , 75 , 76 ] 
 | 
 
 
 [ 77 , 78 , 79 , 80 , 81 , 82 , 83 , 84 , 85 , 86 , 87 , 88 , 89 , 90 , 69 , 91 , 92 , 93 , 94 , 95 , 96 , 97 , 98 , 99 , 100 , 101 , 102 , 103 , 104 , 105 , 106 , 107 , 108 , 109 , 110 , 111 , 112 ] 
 | 
 
 
 [ 77 , 66 , 113 , 114 , 115 , 116 ] [ 81 , 117 , 118 , 68 , 119 , 120 , 121 , 122 , 123 , 124 , 125 , 126 , 127 , 128 , 129 , 130 , 131 , 132 , 133 , 134 , 135 , 136 , 70 , 137 , 138 , 139 , 71 , 140 , 141 , 142 , 143 , 144 , 145 , 72 , 146 , 147 , 148 , 149 , 150 , 151 ] [ 152 ] [ 73 , 153 , 154 , 155 , 156 , 157 , 158 , 159 , 160 , 161 , 162 , 163 , 164 , 165 , 166 , 167 , 108 , 168 , 169 , 170 , 171 , 172 , 74 , 173 , 174 , 175 , 176 , 110 , 75 , 111 , 72 , 177 , 178 , 179 , 180 ] 
 | 

 
 
 
 Cooperative 
 | 
 | 
 
 
 [ 181 ] 
 | 
 
 
 [ 182 , 183 , 184 ] 
 | 

 

 Table I: Example of papers with different types of localization and localization levels. Some papers present different levels of localization. 
 
 
 Active localization is the method most often used in ML-based localization due to its relatively simple setup and wide range of applications. The capabilities of the target device play a role in shaping the localization problem.

 
 • 
 
 In cellular systems, BSs are typically the anchors, and UEs the agents (targets) of the localization. A variety of localization methods, some of them explicitly supported in the standards, have been developed, see Sec. IV. Most of these methods operate in the downlink, i.e., the BSs send out localization signals, and the UE determines its location from the received signals - this is in particular important for privacy reasons. The accuracy can be improved by
acquiring side information using the UEs’ built-in sensors.

 

 • 
 
 Wireless LANs (WiFi) are fairly similar to cellular localization, with Access Points (APs) taking the role of BSs, and stations (STAs) taking the role of UEs. The similarity is most pronounced in enterprise networks, where multiple APs are under the control of the network and can contribute to the localization in a coordinated way. In home settings, often only a single AP is available that can coordinate with the STA, while the other APs are not under the control of the same network/user; signals from those act as interference for communications though they might still be useful for localization purposes, e.g., for fingerprinting.

 

 • 
 
 Wireless sensor networks (WSNs) have gained more attention over the past few years due to the proliferation of the Internet of Things (IoT). Sensor nodes have limited processing capabilities. Therefore, in WSNs, active localization based on uplink transmission is preferred, so that the computations necessary for the localization can be done at the BSs, the gateway or other elements of the infrastructure.

 

 • 
 
 In contrast to traditional WSNs, which tend to have low mobility,
vehicles are usually in highly dynamic environments. Localization there can be done with Road Side Units (RSU). Modern vehicles are usually equipped with many sensors and cameras that can be used as side information. For autonomous vehicles, it is envisioned to achieve reliable sub-meter localization accuracy, which could be difficult with GNSS-only systems.

 

 • 
 
 In passive localization, static transceivers could be set such that they sense the changes in the environment. The ML solution can be used to interpret such changes into a localization. This type of localization can be used for surveillance, occupancy detection, gesture detection, and object counting, see, e.g., [ 185 ] . The difficulty of such problems lies in differentiation between target and non-target objects.

 

 • 
 
 Targets cooperation should usually improve the localization accuracy [ 186 ] . It is a good approach particularly in infrastructure-less or highly dynamic environments. Overall, cooperative localization has received the least attention in ML-based solution.

 

 
 
 
 
 

## IV Features 

 
 Although localization has a wide range of applications and has drawn huge commercial interest, standalone localization systems - with the notable exception of GNSS - are rare. Rather, existing wireless communications systems are adopted to provide also localization information, as a supplementary service. This minimizes the setup cost and increases the deployment speed, but on the other hand creates certain limitations on accessible features. In this section, we first introduce the basic wireless features that are used in localization. We describe how these features are used in ”classical” localization systems, and also survey which papers use them for
ML-based localization. We then discuss their availability in different wireless standards and technologies that have been used thus far. We also summarize some of the used side information along with the wireless signals, and some of the methods for data transformation. A corresponding classification of papers is provided in tables III and II .

 
 

### IV-A Feature Type 

 
 As introduced earlier, an ML-based localization solution is a mapping from the observed features 𝒙 \boldsymbol{x} to the location y y 

 

 
 | 
 y = f ⁡ ( 𝒙 ) . \displaystyle y=f(\boldsymbol{x})\penalty\ \penalty\ \penalty\ . | 
 | 
 (12) | 
 

 Here, y y could be any system defined location output, such as class (e.g., room or RP number) or coordinates (e.g., 3-dimensional coordinates). The input to the ML solution, i.e., the features 𝒙 \boldsymbol{x} , are a design choice that largely depends on the accessibility of such data, usually constrained by the system, and the admissible complexity of the solution. Similarly, the choice of the features used for classical localization algorithms is limited by what is available in particular systems.

 
 

#### IV-A 1 Signal Strength

 
 The simplest and most fundamental feature obtained by a receiver is the receive power, which we define here as the total power received over the bandwidth of the system, P RX ​ ( t ) = P TX ​ ( t ) ​ ∫ | H ⁡ ( t , f ) | 2 ​ 𝑑 f P_{\rm RX}(t)=P_{\rm TX}(t)\int|H(t,f)|^{2}df . 7 7 
 7 
 
 
 
 Needless to say, the integration becomes summation in real system as discrete samples are available. 
The receive power is also often referred to as Received Signal Strength (RSS). Related to it is the Received Signal Strength Indicator (RSSI), which is a – usually vendor specific – quantization of the RSS value. In many cases, only this RSSI is available to localization system, in particular (i) if the information is measured at the UE in a cellular system, and needs to be fed back to the BS for localization, or (ii) the localization system does not have access to the actually received signals, but only the system parameters available via the APIs; this situation occurs when the localization system manufacturer and the chip manufacturer are different. However, in the remainder of this paper, we do not distinguish between receive power, pathloss, and RSSI anymore.

 
 
 If the transmit power is known, this allows to deduce the corresponding (wideband) pathloss. This requires that either the TX does not employ power control, or that the power control settings are known to the RX. For instance, in LTE, the Reference Signal Received Power (RSRP) is the time-averaged received signal of all reference signals from the serving BS. Reference Signal Received Quality (RSRQ), as the name suggests, indicates the quality of the received signal from the serving BS; some of these measurements are reported to the BS, which could be used for localization as used in [ 187 , 188 , 189 , 190 , 191 ] .

 
 
 The RSS information can be used in one of the following two ways: (i) when the channel model is known, RSS can be mapped to distance. Thus localization could be achieved with trilateration with at least three anchors. In pure LOS scenarios, the received power maps to the distance by Friis’ law

 

 
 | 
 P RX = P TX ​ | G TX | 2 ​ | G RX | 2 ​ ( λ w 4 ​ π ​ d ) 2 P_{\rm RX}=P_{\rm TX}|G_{\rm TX}|^{2}|G_{\rm RX}|^{2}\left(\frac{\lambda_{w}}{4\pi d}\right)^{2} | 
 | 
 (13) | 
 

 where λ w \lambda_{w} is the wavelength.
However, in most practical situations the applicable channel model is unknown, making the distance inference prone to errors (note that famous statistical channel models for the received power, such as the Okomura-Hata model, do not constitute a good basis for distance inference because they are averaged over measurements in a large number of environments); even if it were known, the fading variations make a mapping from RSS to distance almost impossible. (ii) RSS can be used as the basis of fingerprinting, in particular when the RSS from several BSs to the target device provides unique fingerprint points. The importance of RSS lies in the fact that it is widely available in most systems and easy to acquire, while the more detailed channel state information might not be always available, see below.

 
 
 Due to multipath, shadowing, as well as hardware impairments, and variations between different devices, the value of the RSSI could vary significantly between different measurements. Thus many works suggest recording the signal over a time window. To overcome the signal variation, many of solutions take several time measurements in the training and testing data, measurements with different devices, e.g., [ 192 , 193 , 194 , 195 ] , posture or directions e.g., [ 162 , 195 , 196 ] . Other solutions apply pre-processing techniques to denoise and stabilize the values.

 
 
 

#### IV-A 2 Time of Arrival

 
 Time of arrival (ToA), is one of the basic quantities that have been used in standard localization systems (e.g., GNSS systems). In pure LOS (free-space) channel, the ToA can be converted to the distance between TX and RX. In a multipath channel, the ToA usually refers to the time of the first detectable MPC. The ToA estimation process usually relies on the TX sending out a training signal at a pre-determined time, from which the RX determines the impulse response. The largest/only (in the case of free-space propagation) or first (in the case of multipath) peak of the determined impulse response is used to determine the pseudo-range, i.e., the runlength (in seconds) between the TX and RX.

 
 
 While determination of the ToA sounds simple in principle, there are a number of problems that arise both from hardware imperfections and the characteristics of the wireless propagation channel. First of all, the above description requires TX and RX to use clocks that are accurately synchronized to each other, which might be difficult to achieve in practice. The requirement can be circumvented by not measuring the runtime from TX to RX, but rather using a two-way signal exchange to determine the round-trip time from TX to RX and back, thus eliminating constant offsets in the clock times. Possible clock drift can be eliminated by multi-packet message exchange [ 197 ] . An alternative approach is the use of Time Difference of Arrival (TDoA), where only the clocks of the different BSs are synchronized, and the localization determines the difference of the pseudoranges from UE to the different BSs, so that the timing offset of the UE clock cancels out; the same principle is also used in GNSSs.

 
 
 Even with perfect clock synchronization, errors do occur due to noise. Assuming Additive White Gaussian Noise (AWGN), the amount of ranging error can be bounded by the Cramer Rao Lower Bound (CRLB)

 

 
 | 
 v ​ a ​ r ​ ( d ^ ) ≥ c 0 2 8 ​ π ​ γ ​ β 2 var(\widehat{d})\geq\frac{c_{0}^{2}}{8\pi\gamma\beta^{2}} | 
 | 
 (14) | 
 

 where c 0 c_{0} is the speed of light, γ \gamma is the Signal to Noise Ratio (SNR), and β \beta is the effective
bandwidth

 

 
 | 
 β = [ ∫ f 2 ​ | S ⁡ ( f ) | 2 ​ 𝑑 f ∫ | S ⁡ ( f ) | 2 ​ 𝑑 f ] 1 / 2 \beta=\left[\frac{\int f^{2}|S(f)|^{2}df}{\int|S(f)|^{2}df}\right]^{1/2} | 
 | 
 (15) | 
 

 with S ⁡ ( f ) S(f) the amplitude spectrum of the signal.
While this bound mainly depends on SNR and carrier frequency f c f_{c} , it is worth noting that it only holds at very high SNR; at intermediate SNRs, the Barankin bound, which is
larger by a factor 12 ​ ( f c / B ) 2 12(f_{c}/B)^{2} than the CRLB, and thus is proportional to the bandwidth , provides a better approximation. This is intuitive, as a finite bandwidth leads to a ”smearing out” of the impulse response, and thus greater difficulty in determining where the true peak is.

 
 
 ToA based methods obviously suffer from problems when the LOS to/from one or more of the anchors is blocked. In a pure LOS situation, this means that no signal (or a signal with insufficient SNR) is received; this situation often occurs for GNSS systems, e.g., in street canyons. Consequently no measurement result is available, which may (or may not) prevent localization by trilateration as discussed below. In multi-path channels, a blocked LOS will often still result in a pseudorange estimate, since the time of arrival of a reflected MPC might be interpreted as a pseudorange, leading to a positive bias of the range estimate. While in some situations having such an estimate might be useful, in other cases it can actually be worse than having no estimate at all, because erroneous information is entered into the trilateration. It is thus important to identify whether a blocked-LOS situation occurs or not, and take this information into account in the trilateration algorithm. A variety of methods for identification of blocked-LOS have been established in the literature, see [ 198 ] for a survey. Besides ”classical” methods for identification of blocked LOS, ML-based methods have been suggested, e.g., [ 199 , 50 ] .

 
 
 The amount of range error that occurs in a blocked-LOS situation depends both on the environment and the particular algorithm used for ToA determination. Firstly, there are some environments in which the delay between the LOS pseudorange and the pseudorange of the first identifiable component is very large, e.g., when a building blocks the connection between TX and RX [ 47 ] . There are also propagation channels with a ”soft onset”, meaning that the first path is not the strongest, e.g. [ 46 ] . In such a situation, the algorithm that determines the range, and the criterion for identification of the ToA become important. For example, detection of the strongest peak in the impulse response can give much larger errors than identification of the first path. Various methods for finding the first path have been suggested, such as ”search forward” from the nominal ”zero delay” of the arriving signal, or ”search backwards” from the time of arrival from the strongest path [ 200 ] . Finally, note that the pseudorange is also increased when a LOS has to propagate through a dielectric material, since the group velocity in such a material is slower than in air.

 
 
 The accuracy of the ToA estimation is also reduced due to finite bandwidth of the system. Firstly, the bandwidth reduces the achievable accuracy in pure LOS scenarios due to the ”smearing out” of the received location signal, which makes finding the peak more difficult. Secondly, in a multipath environment, several MPCs may fall into a resolvable bandwidth and thus superpose constructively or destructively. Thus, the location of the maximum in the impulse response may change when the UE or scatterers move - even when the movement is so small that the delays of the MPCs themselves do not change.

 
 
 Several ML based solutions try to identify the LOS scenarios, e.g., [ 80 , 121 , 157 ] , other have ML solutions that estimate the offset and provide mitigation [ 201 , 83 , 49 ] . To relax the synchronisation constraint, TDoA metrics have been used in the literature [ 121 , 183 , 202 , 203 , 204 ] .

 
 
 

#### IV-A 3 Angle of Arrival

 
 Another classical way of determining the location of an object is triangulation, based on the AoA at (at least) two anchors. Thus, the AoA can be an important feature for localization algorithms. Similar to ToA, also AoA suffers from practical difficulties.

 
 
 First and foremost, determination of the AoA requires real or virtual antenna arrays. The former case implies a physical array with multiple antenna elements that receive the signal (quasi-)simultaneously. These arrays need to fulfill certain conditions in terms of antenna spacing (has to be smaller than half a wavelength to provide unambiguous angles) as well as calibration (the relationship of the complex antenna patterns at the different antenna elements needs to be accurately known). In the virtual array case, a single antenna is sufficient, which is either moved to different locations, or pointed into different directions, at different times - synthetic aperture radar is a famous example of such an approach. However, in this case, the movement of the device creating the virtual array needs to be precisely known.

 
 
 The antenna pattern can usually be determined through calibration of the device. However, the presence of conducting or dielectric objects near the array leads to pattern distortion. Thus, a device being held by a user can have a very different pattern than the calibration result obtained without that user. As a consequence AoA tends to be used mainly at the BSs.

 
 
 A further challenge is the limited resolution. If an analysis is done by Bartlett beamforming (Fourier techniques), then the resolution is limited by the aperture of the antenna array (similar to the bandwidth limiting the ToA resolution). If High Resolution Parameter Estimation (HRPE) techniques such as MUSIC, ESPRIT, or SAGE, are applied, higher resolution can be achieved, though these algorithms are sensitive to errors in the calibration, and also require high computational complexity [ 36 ] .

 
 
 Just like for ToA, the biggest challenges arise when the LOS is blocked. In this case, the system might determine the direction of the strongest AoA, but this might not be identical to the LOS direction. And in contrast to the ToA case, where the ”earliest significant” MPC is often associated with an attenuated LOS, no such identification is possible for the AoA case. For these reasons, it is rare to use only AoA for determination of location; rather, it is used to augment estimations based on other features. This is particularly true for classical localization approaches.

 
 
 To overcome the high demands on signal processing and calibration, many ML papers use coarse estimates of AoA. For example, [ 205 ] noticed that the phase difference between antenna pairs is correlated with the true AoA, Ref. [ 206 ] use the recorded directional RSSI value, with simplified angle calculation method in [ 207 ] , to estimate the AoA. Note that with a large number of antennas, finer estimates of AoA become easier to obtain.

 
 
 

#### IV-A 4 Channel State Information

 
 Channel State Information (CSI) is often defined as the complex value of the received signal, i.e., amplitude and phase values, for all subcarriers, 8 8 
 8 
 
 
 
 We assume here and henceforth an OFDM system, since it is by far the most popular implementation of wideband transmission. Equivalently, the complex samples of the impulse response, spaced at most at the Nyquist sampling rate, can be used. at all antenna elements. This CSI is clearly the most comprehensive ”raw” information obtained by the system; the other features mentioned above, such as RSSI, ToA, and AoA are compressed, derived versions of it. CSI can thus serve as the basis of compressed information, possibly expanding the number of derived features. Alternatively, CSI can be used directly in fingerprinting. The operating principle of CSI-based fingerprinting is the same as for RSSI-based fingerprinting; yet much more accurate fingerprinting can be established because of the richer information contained in the CSI. For example, it is possible to obtain localization from the CSI of the link between target and a single anchor, as was done in [ 60 , 77 , 174 , 57 , 58 , 208 , 209 ] . Note that such an approach is not practical for RSSI-based fingerprinting because the same RSSI can occur at multiple locations around an anchor; after all RSSI is only a single scalar. Conversely, it is very unlikely that two locations have the same CFR, since this would require that the complex channel response is the same at all the subcarriers, i.e., agreement of all entries of a complex vector . The main challenges of using CSI are (i) complexity of using such rich information, and (ii) availability of the information at the APIs used by the localization system.

 
 
 
 

### IV-B Wireless Technology 

 
 Advanced wireless technologies, in particular multi-antenna systems and ultra-wideband systems, are especially helpful in localization. While they can be interpreted as simply providing particular features (direction, ToA) with increased accuracy, we briefly survey their background and their application in ML-based localization.

 
 

#### IV-B 1 Multi-Antenna Systems

 
 When the TX and/or RX are equipped with multiple antennas, the system can collect several copies of the wireless signal. For localization, it can be used to: (i) Improve the estimate of the features by enhancing the signal to interference and noise ratio (SINR) through beamforming and interference nulling [ 36 ] . (ii) Improve the resolution of AoA estimation. (iii) Enhance the separability of the fingerprints, by providing a larger dimensionality of the fingerprint. Multi-antenna systems are often called multi-input multi-output (MIMO) systems - in an abuse of notation, this term is even used when multiple antenna elements are only at one link end.

 
 
 In the ML-based localization literature, different methods were proposed to utilize multiple antennas. Several papers use the difference of values between antennas pairs to create stable features, e.g., phase difference [ 210 , 211 ] , as it eliminates the constants that are common to the antennas, making the features less sensitive to specific device models. Furthermore, large arrays may allow for sparse representation of the channel in the angle domain.
The advent of 5G has triggered interest in large antenna systems. Thus, many recent works proposed localization for massive MIMO [ 212 , 213 , 214 , 215 , 216 , 217 , 208 , 218 , 219 , 209 , 220 ] .

 
 
 

#### IV-B 2 Ultra Wide Band (UWB) Systems

 
 Time resolution increases with the bandwidth of the wireless system, which can allow for finer ToA estimates. It has further the advantage of combating the small scale fading, as fewer MPCs might fall into the same delay bin. UWB systems are systems with at least 500 MHz bandwidth; they are permitted (in the US) to operate in the frequency range 3.1 − 10.6 3.1-10.6 GHz subject to constraints of the power spectral density [ 221 ] . Systems with similarly large, or even larger, bandwidths can also exist at higher carrier frequencies, such as the 60 GHz unlicensed band. The large bandwidth leads to resolvable delay bins that are fractions of nanoseconds, thus providing a localization accuracy in the order of centimeters [ 200 , 2 ] .

 
 
 These advantages have also motivated the use of UWB in ML-based localization, e.g., [ 222 , 202 , 223 , 224 ] , It is possible to acquire and use directly the CIR as a feature, e.g., see [ 225 ] . It is also possible to estimate with improved accuracy a number of compressed channel parameters, such as Power Delay Profile (PDP), delay spread, and other quantities that could capture the structure of the environment. For instance, [ 224 ] uses delay spread, rise time, mean excess delay (among others) for ranging error mitigation in UWB systems. Since mmWave systems are expected to use multi-antennas (to compensate for the increased path loss at higher frequencies), recent works explore suitable features, e.g., [ 180 ] uses the set of beamformed signals and PDP as input to a DL algorithm. In [ 183 ] , the authors use hybrid delay and angle measurements in a cooperative mmWave system.

 
 
 

#### IV-B 3 Multi-Carrier and Multi-Band Systems

 
 As we have seen in Sec. II.B, impulse response and CFR of a channel are equivalent, and related through a simple FT. Yet, which of those to use can still make a difference in implementations of ML algorithms. Most practical cellular and WiFi systems are multi-carrier signals, so that they naturally measure the CFR, or more precisely, the samples of the CFR at discrete frequencies, the subcarrier frequencies used in the multicarrier signaling.
When CSI at each sub-carrier is available, CFR is usually used as feature [ 212 , 226 , 208 ] . Note that in classical localization algorithms, FT and use of the impulse response for localization is more common, even when the system is based on multicarrier communications.

 
 
 On a larger scale, wireless systems might use different frequency bands to utilize the inherent propagation advantages over different bands or because they might simply coexist, e.g., 2.4GHz and 5GHz in WiFi or centimeter-wave (cmWave) and mmWave bands in 5G networks. Features over these bands can be independent or complement each other, e.g., wide coverage in the cmWave band, and better delay resolution in the mmWave band. Some recent works started to investigate such multi-band systems, such as 2.4 2.4 GHz and 5 5 GHz in [ 196 , 227 ] . Ref [ 227 ] derived features from RSSI at 5 5 GHz for LOS identification, then uses RSSI in 5 5 GHz and 2.4 2.4 GHz for LOS and NLOS localization, respectively.

 
 
 
 
 
 | 
 
 
 Papers 
 | 

 
 
 
 
 
 OFDM 
 | 
 
 
 [ 84 , 228 , 229 , 230 , 231 , 232 , 233 , 234 , 226 , 235 , 211 , 236 , 190 , 237 , 212 , 214 , 215 , 216 , 217 , 209 , 238 , 239 , 61 ] 
 | 

 
 
 
 MIMO 
 | 
 
 
 [ 240 ] [ 228 , 230 , 231 , 232 , 205 , 233 , 234 , 226 , 235 , 211 , 236 , 190 , 237 , 217 , 241 , 238 , 206 , 239 , 242 , 61 ] 
 | 

 
 
 
 Massive MIMO 
 | 
 
 
 [ 113 , 174 , 213 , 212 , 214 , 215 , 216 , 208 , 218 , 209 , 243 ] 
 | 

 
 
 
 Distributed MIMO 
 | 
 
 
 [ 113 , 213 , 218 , 219 ] 
 | 

 
 
 
 UWB 
 | 
 
 
 [ 80 , 83 , 141 , 244 , 157 , 49 , 202 , 222 , 223 ] 
 | 

 

 Table II: Technologies used in a sample of papers. 
 
 
 
 

### IV-C Standards Type 

 
 Localization in wireless systems can be either based on signals and protocols that are explicitly designed for the purpose of localization, or they can make use of ”incidental” signals of systems designed for communications.
When the localization system is built based on a wireless communication standard, the available features are restricted by the specifications of the underlying standard, e.g., the bandwidth, the central carrier frequency and the maximum supported number of antennas. Furthermore, the communication protocols could also restrict the access to certain features, e.g.,, does the target have to be in ”connected mode” to observe the feature? This is important as it limits the number of usable anchors. In this subsection we provide a brief review of some of the standards commonly used in ML-based solution; details can be found in number of other dedicated paper [ 5 , 10 , 12 , 245 ] .

 
 
 The most famous example of an explicit localization protocol is GPS - actually GNSS systems have as their only purpose localization, so that clearly their protocols and signals are designed for localization. The basic principle of GNSS is TDOA; the particulars of the localization signals and synchronization procedures can be found, e.g., in [ 1 ] .

 
 
 Another important explicit ranging protocol is used in LTE. There, different BSs send out ranging signals (either on demand, or at regular intervals) that can be used by the UE to determine its location by trilateration. Those ranging signals are similar to the ”reference signals” (pilot signals) used for channel estimation. The difference is that during the time that one BS sends out a localization signal, the surrounding BSs are silent, thus drastically improving the SINR and consequently the precision of the ranging. However, ranging in LTE can also use other, non-standardized, procedures, such as CSI in either uplink or downlink (which is acquired as a matter of course in the normal operation of the system). Localization can be based either on the directly measured CSI, or (When that is not available at the APIs), by using compressed CSI such as RSRP and RSRQ of the serving BS and possibly one or more of the neighbor BSs, as conveyed in the measurements reports (MRs) that are sent by the UEs either periodically or triggered by certain events or procedures. A number of proposed solutions use MRs for ML-based localization, such as radio map construction or channel model based localization. LTE cellular networks have been used in number of ML solutions, such as [ 246 , 191 , 189 , 187 , 247 ] .

 
 
 Another standard that has an explicit protocol for ranging is IEEE 802.15.4a, which is based on UWB signalling. The ranging signal there is a pseudorandom sequence with special correlation properties [ 248 ] . It has been widely used for deterministic localization experiments, but is not widely used in ML solutions, possibly due to the lack of extensive deployments.

 
 
 Besides cellular systems, WiFi (802.11) is the most widely used standard for data transmission. However, until recently there had not been an explicit localization protocol. A key challenge is that neither the location of the APs is precisely known, nor are the APs generally synchronized to each other. Furthermore, in home networks, only a single AP is under the control of a particular network, so that data packets to/from other APs, which are encrypted, cannot be used to help with the localization. The only contribution that APs of other users can provide are the beacon signals, which can be detected by a UE without being authorized to log in and decrypt. Note that since the deployment is usually indoor, and the LOS is blocked, determination of the distance based on the received signal power is not easily possible. Yet the beacon signals provide RSSI that can be used for fingerprinting and/or ML solutions.

 
 
 WiFi based ML localization tends to use RSSI measurements to the serving AP, as well as RSSI to other APs as input features due to the easy availability. However, since many WiFi releases, such as IEEE802.11n, use OFDM and MIMO technologies, a number of recent work started to use CSI (with the serving AP) over different sub-carriers and antennas. Enhancements to timing measurements were introduced in 802.11-2016 (IEEE 802.11 REVmc) through Fine Time Measurements (FTM) protocol, this allow to use better ToA estimates as features for localization which is used, e.g., in [ 194 ] .

 
 
 With the growing commercial interest in proximity based Services, localization using Bluetooth Low Energy (BLE) has attracted considerable attention [ 245 ] , especially after major companies such as Apple and Google released their BLE protocols, iBeacon and Eddystone, respectively. Although it is possible to locate the target via BLE RSSI readings (as in WiFi), due to practical reasons, the focus was on ranging, as it was also encouraged by iBeacon protocol, where the goal is to identify the proximity of the target.

 
 
 In the literature, other standards have been used as well for localization, such as Zigbee [ 131 , 95 , 143 , 145 ] and FM signals [ 129 ] . However, these systems are not as popular as the ones above, possibly due to their smaller installed userbase. Note that there are many other emerging standards and protocol that could be potentially used for localization, such as LoRaWAN and SigFox for IoT (see [ 12 , 5 ] for good surveys), and next generation wireless standards, 5 5 G and beyond.

 
 
 
 
 
 | 
 
 
 WiFi 
 | 
 
 
 Cellular 
 | 
 
 
 BLE 
 | 
 
 
 WSN § 
 | 
 
 
 Other 
 | 

 
 
 
 
 
 RSSI 
 | 
 
 
 [ 181 , 79 , 66 , 114 , 81 , 68 , 124 , 125 , 126 , 184 , 86 , 132 , 88 , 134 , 89 , 69 , 92 , 70 , 53 , 96 , 137 , 139 , 71 , 140 , 97 , 142 , 72 , 146 , 147 , 98 , 148 , 100 , 101 , 149 , 150 , 249 , 250 , 152 , 251 , 103 , 73 , 153 , 154 , 252 , 156 , 159 , 104 , 160 , 161 , 162 , 163 , 105 , 166 , 167 , 107 , 108 , 168 , 171 , 172 , 253 , 74 , 173 , 109 , 175 , 110 , 75 , 111 , 112 , 254 , 72 , 177 , 255 , 256 , 196 , 257 , 233 , 258 , 259 , 260 , 261 , 262 , 263 , 264 , 265 , 266 , 267 , 268 , 269 , 270 , 271 , 272 ] [ 273 , 274 , 275 , 276 , 277 , 278 , 279 , 280 , 281 ] 
 | 
 
 
 [ 113 , 115 , 134 , 90 , 94 , 149 , 76 , 65 , 179 , 282 , 188 , 192 , 187 , 247 , 283 , 191 ] 
 | 
 
 
 [ 85 , 127 , 128 , 284 , 106 , 111 , 285 , 258 , 286 , 287 ] 
 | 
 
 
 [ 119 , 120 , 143 , 145 , 102 , 64 , 288 , 176 , 178 , 289 , 290 , 291 , 78 ] 
 | 
 
 
 [ 182 , 82 , 84 , 129 , 56 , 87 , 131 , 91 , 136 , 95 , 138 , 59 , 244 , 284 , 165 , 62 , 292 , 293 , 294 , 295 , 296 ] 
 | 

 
 
 
 CSI 
 | 
 
 
 [ 54 , 55 , 135 , 57 , 58 , 93 , 99 , 60 , 77 , 172 , 61 , 297 , 228 , 230 , 231 , 232 , 205 , 233 , 234 , 298 , 226 , 299 , 235 , 211 , 236 , 237 , 300 , 206 ] 
 | 
 
 
 [ 118 , 185 , 216 ] 
 | 
 | 
 
 
 [ 63 ] 
 | 
 
 
 [ 133 , 174 , 49 , 240 , 301 , 238 ] 
 | 

 
 
 
 ToA 
 | 
 
 
 [ 250 , 302 ] 
 | 
 | 
 | 
 | 
 
 
 [ 80 , 141 , 244 , 165 , 49 , 203 ] 
 | 

 
 
 
 Other 
 | 
 
 
 [ 303 , 79 , 121 , 123 , 144 , 155 , 158 , 229 ] 
 | 
 
 
 [ 67 , 52 , 116 , 117 , 121 , 94 , 190 ] 
 | 
 | 
 
 
 [ 83 ] 
 | 
 
 
 [ 122 , 183 , 87 , 157 , 164 , 170 , 180 , 202 , 225 ] 
 | 

 

 Table III: Technology and basic features. § Note that for WSN a number of works use unsupervised learning methods based on a presumed knowledge of the distance between the nodes, which can typically be estimated using ToA or RSSI, we here limit the discussion of these solution due to these implicit assumptions about the RF signal and presence of dedicated survey paper [ 21 ] (see Sec. V-C for more details). 
 
 
 

### IV-D Side Information 

 
 One advantage of using ML is its ability to integrate efficiently different, and seemingly non-homogeneous, sets of features in the solutions. For many localization problems, it is possible to acquire additional information besides the RF signal. For instance, modern cellphones are equipped with light sensor, acoustic sensors, accelerometers, gyroscopes, and magnetometers. Some of these sensors can be integrated into what is referred to as the Inertial measurement unit (IMU). Using these measurements, it is possible to detect the motion, identify the orientation, or environment change, which can help in the localization and tracking process.

 
 
 Maps and environment structures, such as floor plans, have been used as side information. It can help imposing physical constraints on the solution, such as walls and entrances. This results in realistic and smooth trajectories for tracking applications. Maps are usually integrated using map matching algorithms, e.g., using HMMs, where the transition between the hidden states are governed by the map constraints, see for instance [ 304 ] .
Images can be used for localization, e.g., using photos of local ”landmarks” as RPs. Note that each of the above can be used as the only feature for localization. However, in this survey, we highlight solutions that use them along with RF signals; a list of such ML-localization papers is presented in table IV .

 
 
 
 
 
 
 
 Maps 
 | 
 
 
 Magnetic field 
 | 
 
 
 Accelerometer 
 | 

 
 
 
 [ 122 , 262 , 275 , 277 , 279 , 72 ] 
 | 
 
 
 [ 125 , 96 , 184 , 89 , 169 , 111 , 179 , 258 , 278 , 305 , 306 ] 
 | 
 
 
 [ 68 , 120 , 125 , 184 , 89 , 111 , 72 , 179 , 278 , 279 ] 
 | 

 

 Table IV: Side information 
 
 
 

### IV-E Feature Representation 

 
 As we will discuss below, in some systems, it is possible to collect several values of the basic features, e.g., RSSI values from nearby BSs/APs. Several ML-based localization solutions use statistical quantities, such as the mean, variance, kurtosis, and skewness as compact feature representation. These quantities are usually more stable, but they reduce the granularity of the observations. Thus they can be used for coarse localization or used jointly with other features.

 
 
 Dimensionality reduction techniques have been widely used, including linear methods such as PCA and Linear Discriminant Analysis (LDA) (a supervised dimensionality reduction technique) and several non-linear techniques that we discuss as part of the later sections. One goal of these techniques is to provide compact and robust representations of the features without the loss of useful data, thus making learning easier and more stable.

 
 
 A number of solutions use signal processing techniques to transform the features to other domains. The goal is to utilize some advantageous properties in the new domain. For instance, the FT is used to transfer the directional CFR acquired in an OFDM-MIMO system from the frequency-antenna domain to the delay-angle domain in the form of CIR, which has been observed to improve localization performance. One easy explanation is that it is straightforward to locate the target when the delay and the AoA of the LOS component are identified. These transformations have been used in several works, e.g., [ 212 , 214 , 208 ] . Wavelet transform is another method that has been used to denoise the received signal and to provide higher dimensional representation, e.g., from 1D time series vector to 2D time-frequency ”images”, as in [ 307 , 271 ] where they use the wavelet to transform the RSSI readings in WiFi system.

 
 
 Another group of papers aims to reduce the correlation between the acquired features, mainly RSSI, through AP selection, e.g., [ 308 , 182 , 86 , 309 ] . Examples of other transformations include using visibility graphs, [ 310 ] , to transform a series of CSI values on different sub-carriers into a network representation that could
reveal the internal relationship of data and improve classification results as used in [ 60 ] .

 
 
 

### IV-F Features Utilization in Conventional and ML-based Localization Systems 

 
 Before going into the details of ML-based solutions, it is useful to review how conventional localization algorithms use the above features, and where the ML methods were introduced in a number of ML-based localization. The most common principles are trilateration/triangulation and fingerprinting (proximity and direct localization methods can be viewed as special cases of the two). Trilateration uses estimates of the distance between anchors and target to determine the location of the target. In the case of active localization with ToA estimation, each range estimate corresponds to a circle; for TDoA to a hyperbola, and for passive estimation with ToA to an ellipse. The intersection of those curves then provides the location estimate. The accuracy of the location is determined by the accuracy of the range estimate as well as the Geometric Dilution of Precision (GDOP) due to ”grazing” intersection of the curves corresponding to the different anchors. When AoA is available at a particular anchor, it provides additional information, namely that the target must lie on the line corresponding to this angle. Taking into account the uncertainty of the range estimate due to noise, various linear and nonlinear techniques have been developed for the computation of the most likely intersection point of the circles [ 1 ] .

 
 
 Naive application of this approach can lead to significant errors, in particular in the presence of LOS blockage and/or multi-path. The (positive) bias of the range estimates from NLOS links leads to circles that are larger than the true distance between anchor and agents. Consequently, the circles corresponding to the different anchors might not intersect in a single point at all. A variety of methods have been developed to perform localization under such circumstances; for example POCS [ 311 ] , and approaches based on consistency of the estimated solutions [ 312 ] . A further improvement can be achieved by employing ”soft information” [ 313 ] . While most trilateration approaches find the mean and variance of the range estimate, and combine those, soft information uses the probability density function or log-likelihood of the range estimate as the input, and from that determines the most likely location. In the presence of non-Gaussian, asymmetrical ranging errors, such as occur in the presence of blockage and multi-path, this can lead to a considerable improvement of the accuracy.

 
 
 Many recent localization systems utilize ML to improve trilateration/triangulation systems, e.g., to predict the type of environment (e.g., indoor vs outdoor [ 187 , 179 ] , LOS vs NLOS [ 157 , 121 , 80 , 49 , 223 , 50 ] ), or to correct the bias in the range estimates [ 224 , 225 , 201 ] , or for a direct range estimation [ 244 , 314 ] .

 
 
 The fingerprinting can be summarized as follows. The system operates in two phases: an offline and an online phases. During the offline phase, the environment is surveyed at predefined locations (RPs) and certain features of the RF signal are recorded; location of the RPs and the recorded features at those points are then stored in a database (radio-map). The majority of papers are based on RSSI signals from multiple APs, however, as discussed above, recent methods start to employ CSI as well. During the online phase, when the location of a target is requested, the system matches the observed features to the ones in the database. Matching is based on deterministic or probabilistic methods. In the latter the system finds the most probable RPs for the given observations. The output of the fingerprinting system (i.e., target’s location), is usually a function of the location of one or more of the matched RPs. The research related to fingerprinting systems revolves around the best methods to construct the radio map, the matching techniques, and final location estimate (i.e., observed feature to location mapping). For instance, one of the main challenges in the practical implementation is the requirement for a fast search for the fingerprint in the database that best matches the current observation. Straightforward linear searches are too computationally intensive in particular when large databases are available; rather a hierarchical search is preferable.

 
 
 Localization usually refers to the determination of the location from observations (features) at a single time instant. However, localization systems often have measurements at multiple time instances available which - when associated with a moving target - allows to track the target trajectory. Such tracking, or in general utilization of previous observations (historical data), is usually done by means of Kalman filters or particle filters [ 1 ] . They predict the change of location in a timestep based on the previous movement, and subsequently correct the estimate from measurements of the new location. Both the prediction and the correction are probabilitistic, and their combination provides an estimate that filters out some of the noise in the measurements.

 
 
 ML has been used in fingerprinting solutions since its infancy, where standard ML methods have been utilized as matching mechanism, e.g., [ 315 , 6 ] . Since then it has been utilized in other aspects as well, for instance for feature extraction and radio-map construction [ 297 , 231 , 103 , 154 , 72 , 91 , 94 , 138 , 142 , 72 ] , radio-map updating [ 181 , 132 , 122 , 85 ] , hierarchical solutions [ 316 , 222 , 151 , 256 , 75 ] , and robust matching [ 79 , 88 , 86 ] . This is expected because fingerprinting systems, similar to ML, are data-driven and both may 9 9 
 9 
 
 
 
 Note that some methods have been proposed to construct the radio-map online, and some ML solution may operate fully online (”online learning”). operate in a training (offline) phase, and an online phase. Nevertheless, ML-based solutions span a wider scope as, for instance, ML could be used as an end-to-end (i.e., direct) localization solution, without the need for an explicit radio-map construction (compare the ”instance learning” to other methods in Sec. II-A ). Tracking problem may utilize ML solutions to assist the traditional tracking methods (e.g., improving Kalman filter based tracking as in [ 120 , 114 , 125 , 70 , 71 , 317 ] ). Furthermore, many methods to track the targets make assumptions about the location evolution process; the advances of recurrent DL methods can potentially eliminate the need for such restrictive assumptions.

 
 
 
 

## V Learning Solutions 

 
 As discussed in Sec. II-A , we can distinguish a number of ML classes:

 
 • 
 
 Supervised Learning: features and labels are present during the training phase.

 

 • 
 
 Semi-Supervised Learning: labels are available for a subset of the features.

 

 • 
 
 Unsupervised Learning: no labels are available.

 

 • 
 
 Other Learning Approaches: labels may be available but training is done differently, for example, in sequential or distributed fashions.

 

 
 We can also distinguish between the standard and the DL approaches. We here refer to approaches that use hierarchical learning structures as DL; with this definition, NNs with a number of hidden layers can be classified as DL. Note that it is not easy to draw a line between the two, thus it is possible to classify some of the reported papers below differently. Furthermore, as discussed in the previous section, probabilistic methods are used for fingerprint matching. Based on the discussion in Sec. II-A , one might view, for instance, all Bayesian methods, mixture models, and HMM as part of the standard ML methods. However, in this survey, we do not focus on those methods as they lie in the gray area between model-based analytical solutions and ML (though we do cover a number of DL probabilistic methods).

 
 
 In this section, we present a representative set of different structures for each of the aforementioned ML approaches, and summarize a number of works that adopted them for localization. In each subsection we order the reviewed papers based on the feature complexity, which is usually related to the technology and the standards.

 
 

### V-A Supervised learning 

 
 In this section, we start by reviewing works that use standard ML algorithms, followed by those using DL algorithms. As highlighted above, the organization is done based on the perspective of the used algorithm; for DL this could provide a glimpse at the considered problem as it influences the DL architecture. For standard ML this is generally not the case; however, we follow the same structure, as it is motivated by the goal of this survey (emphasis on recent ML solutions which tend to be DL based), and to maintain the organizational consistency.

 
 
 
 
 
 | 
 
 
 Supervised 
 | 
 
 
 Semi-Supervised 
 | 
 
 
 Unsupervised 
 | 
 
 
 Other 
 | 

 
 
 
 
 
 Standard ML 
 | 
 
 
 [ 181 , 79 , 54 , 182 , 116 , 80 , 81 , 117 , 118 , 68 , 120 , 121 , 122 , 125 , 83 , 84 , 126 , 183 , 128 , 129 , 184 , 86 , 56 , 87 , 57 , 131 , 132 , 88 , 89 , 69 , 63 , 91 , 92 , 94 , 136 , 70 , 53 , 96 , 137 , 138 , 71 , 140 , 97 , 244 , 142 , 72 , 249 , 102 ] [ 252 , 60 , 156 , 157 , 104 , 160 , 161 , 162 , 165 , 106 , 166 , 108 , 288 , 61 , 109 , 175 , 176 , 110 , 75 , 112 , 254 , 72 , 76 , 178 , 303 , 318 , 303 , 319 , 287 ] 
 | 
 
 
 [ 320 , 66 , 67 , 113 , 114 , 119 , 139 , 146 , 143 ] [ 100 , 65 ] 
 | 
 
 
 [ 85 , 127 , 56 , 95 , 112 ] 
 | 
 
 
 [ 123 , 124 , 98 , 150 ] [ 251 , 161 , 64 , 173 , 179 ] 
 | 

 
 
 
 Convolutional based 
 | 
 
 
 [ 135 , 59 , 147 , 148 , 164 , 105 , 77 , 171 , 172 , 62 , 180 ] 
 | 
 | 
 | 
 | 

 
 
 
 Recurrent 
 | 
 
 
 [ 321 , 272 , 115 , 52 , 117 , 147 , 84 , 133 , 149 , 284 , 153 , 163 , 167 , 169 , 180 ] 
 | 
 | 
 | 
 | 

 
 
 
 Other 
 | 
 
 
 [ 118 , 88 , 58 , 91 , 93 , 70 , 140 , 145 , 99 , 101 , 151 , 250 ] [ 152 , 103 , 73 , 154 , 252 , 155 , 158 , 161 , 185 , 166 , 77 , 168 , 170 , 253 , 74 , 174 , 111 , 177 ] 
 | 
 
 
 [ 67 , 82 , 130 , 90 , 147 ] 
 | 
 | 
 
 
 [ 82 , 103 , 159 , 107 ] 
 | 

 

 Table V: Type of ML model used 
 
 

#### V-A 1 Standard Machine Learning Solutions

 
 In this section we review different localization solutions based on standard ML techniques. For brevity we review representative works that capture different aspects of the research in this area. Note that many research works present several ML algorithms when evaluating their solution’s performance (i.e., use some of them as benchmarks). We here report the main solutions of those papers.

 
 
 K-Nearest Neighbors (KNN): 
In KNN, the coordinate of the target is approximated by the average location of the K K closest RPs in the fingerprint database. When the goal is to identify the region, e.g., the room, the predicted region is the one where the majority of the K K neighbors are located, see Sec. III for more details about these two cases. In Weighted KNN (WKNN) the locations of the K K closest neighbors are given different weights before averaging them, i.e., KNN is used as a fusing technique. Due to this intuitive structure and relative simplicity, KNN has been considered in many localization solutions, especially fingerprint based localization. When using KNN, research has been dedicated to fingerprint database construction methods, adaptively selecting K K , and deriving meaningful similarity metrics. Here we review a few works to highlight these aspects. Other work can be found in various surveys for fingerprints, e.g., [ 11 , 6 ] .

 
 
 A typical early example for this approach is [ 315 ] , where the collected RSSI values are stored to construct the radio map. During the online phase, i.e., the localization phase, the newly observed RSSI value from three indoor APs are compared against the stored RSSI dataset. The comparison is usually done using Euclidean distance,

 
 
 

 
 | 
 ℒ ⁡ ( 𝒙 i , 𝒙 0 ) = ∑ k = 1 N | x i , k − x 0 , k | 2 \mathcal{L}(\boldsymbol{x}_{i},\boldsymbol{x}_{0})=\sqrt{\sum_{k=1}^{N}|x_{i,k}-x_{0,k}|^{2}} | 
 | 
 

 where N N is the number of APs (three in [ 315 ] ) and x i , k {x}_{i,k} and x 0 , k {x}_{0,k} are, respectively, the RSSI value from the k th k^{\rm th} AP at the i th i^{\rm th} RP and the target. 10 10 
 10 
 
 
 
 We stress that x x is the RSSI, and not a location coordinate; this is to stay consistent with Sec. II-A that generally denotes features as x x . The standard KNN uses fixed K K value; however, due to RSS variations or the placement of the RPs (e.g., enforced by the environment structure), using fixed K K might result in large localization estimation errors. To alleviate that, a number of works considered modifications of KNN, e.g., [ 162 , 318 , 322 , 287 , 319 ] . The methods range from sub-selecting some of the K K nearest neighbors as in [ 318 , 319 , 322 , 162 , 323 ] , to modifying the similarity (i.e., the distance) metric [ 287 ] . For instance, the solution in [ 318 ] groups the K K nearest RP neighbors in different clusters (according to their locations), then the final location is the average of the RPs in the ”delegate” cluster. The work in [ 287 ] suggests using both the Euclidean distance and the cosine similarity between the RSSI values as a similarity metric, which they also use to construct the weights for a WKNN.

 
 
 KNN has been also used with different features. In [ 296 ] the authors built a cascaded localization system based on KNN, where a KNN is used to first identify the environment, then for localization they apply KNN with different features (RSSI, the CFR and its auto-correlation function). They note the importance of considering hybrid features in different environments, and that they outperform the RSSI-only approach.

 
 
 The AoA can be used as a feature in MIMO system; for instance [ 303 ] uses AoA to sub-select a group of the K K RPs that satisfy an AoA constraint. Ref. [ 209 ] proposes a fingerprint-based single-site localization method for massive-MIMO OFDM systems. However, storing the raw channel observations and searching through them requires large storage and incurs high computational complexity. The paper proposes an efficient database construction by utilizing the sparse representation of the channel in the angle-delay domain, which can be efficiently compressed. To search over different fingerprints, it uses two levels of fingerprint classification and clustering. The similarity between the observed channel and fingerprints is captured by a proposed joint angle and delay similarity metric that depends on the level of overlap of the scatterers. Finally, localization is performed by applying a WKNN to the K nearest fingerprints, where the weights are functions of the proposed metric that correspond to the AoA and ToA. Along the same lines, [ 208 ] proposes a method to construct the database of the angle-delay domain representation by compressing the database and applying a fast method to retrieve the candidate RPs using hashing algorithms; localization is obtained from a WKNN based on the minimum Euclidean distance of the features.

 
 
 In Ref. [ 184 ] , a crowd-sourcing indoor localization algorithm via an optical camera and orientation sensor on a smartphone is introduced. In this solution fingerprint based localization uses KNN over RSS from WiFi, which is used as a coarse estimate of the location to speed up the search space of images based localization; this can be further constrained with information from orientation sensors.

 
 
 In summary, since WKNN is an intuitive technique, many advanced ML solutions use it as a final step to predict the location, where different methods are proposed to define the ”distance” between the fingerprints and the weights. Methods to calculate the distances and weights include Euclidean distance and Kernel functions, respectively. Additional details can be found in later sections.

 
 
 Kernel Based Methods: 
A kernel function 𝒦 ( . , . ) \mathcal{K}(.,.) captures the similarity between vectors, and thus can be integrated in instant based learning techniques, see Sec. II . One natural way to capture the similarly between two vectors 𝒙 i \boldsymbol{x}_{i} and 𝒙 j \boldsymbol{x}_{j} is based on the distance between them. The Radial Basis Function (RBF) is a kernel that is a function of the distance. An example of that is the Gaussian Kernel

 

 
 | 
 𝒦 ⁡ ( 𝒙 i , 𝒙 j ) = exp − ‖ 𝒙 i − 𝒙 j ‖ 2 2 ​ σ 2 , \displaystyle\mathcal{K}(\boldsymbol{x}^{i},\boldsymbol{x}^{j})=\exp^{-\frac{||\boldsymbol{x}^{i}-\boldsymbol{x}^{j}||^{2}}{2\sigma^{2}}}, | 
 | 
 (16) | 
 

 where σ \sigma controls the speed of similarity decay between 𝒙 i \boldsymbol{x}^{i} and 𝒙 j \boldsymbol{x}^{j} . Other kernels include linear, polynomial, and sigmoid kernels.

 
 
 One of the powerful algorithms that uses kernels is the Support Vector Machine (SVM). It aims at finding the best hyperplane to separate different classes. In particular, the hyperplane is chosen such that it maximizes the margin between the different classes. In SVM the predictions usually appear in the form of inner products between training instances and observed feature; this allows SVM to use the kernels to find efficiently this product in higher (possibly infinite) dimensional space, which can increase the separability of the features, see Sec. II-A .

 
 
 SVM, along with its regression counterpart SV-Regression (SVR), and other kernel methods have been used extensively in localization. In one of the early works that uses kernel based localization, [ 78 ] uses SVM for localization in a WSN that has a few sensors with known locations. Then the sensors measure the RSSI between one another, in order to obtain an initial region classification using an SVM with Gaussian kernel. Finally, finer coordinates are obtained by averaging the centers of the regions each sensor belongs to. Ref. [ 76 ] uses linear and Gaussian kernels for room level localization using cellular RSSI. Ref. [ 249 ] first reduces the dimensionality of the RSSI features through PCA, then localizes the target using SVM with an RBF kernel. In [ 104 ] the authors propose a method to address the diversity of the devices by first sorting the RSSIs from the most reliable APs, then use SVM to classify the target location. Ref. [ 308 ] proposes real time AP selection to minimize the number of correlated APs used, and then uses a kernel as a metric measure to construct a weight based on the similarity between the RSSI observation and the RPs, which is then used to weight the coordinates of the RPs to estimate the location.

 
 
 SVM and other kernel methods have been used with other features as well. In [ 165 ] the authors used ToA and RSSI in Gaussian kernel Ridge regression for localization. Ref. [ 203 ] uses TOA as fingerprints, and proposes a Gaussian Kernel based solution; the proposed approach is insensitive to random synchronization and measurement timing errors. Ref. [ 222 ] proposes a localization solution in UWB systems, by extracting different features from the channel impulse response, such as the energy decay time. The idea is to compress the impulse response and capture different aspects of the environment. The features are then fed to an SVM algorithm for hierarchical region classification. Using CSI of 30 sub-carriers, [ 60 ] constructed a visibility graph that captures the frequency correlations between adjacent sub-carriers, which are used for localization with SVM. CSI based activity recognition and localization using SVM and kernel regression is proposed in [ 229 ] , where the SVM is first used to classify the target in one of the possible activity classes, then the localization is done with the associated localization regression model. In a device-free localization system, [ 61 ] used the CSI in an OFDM-MIMO system with different detecting points to sense the channel and send it to a central sever. Localization with RSSI and accelerometer readings was used in WSN target tracking in [ 120 ] ; the RSSI based kernel method provides a coarse location estimate, which is then utilized along with the accelerometer readings to get the instantaneous localization using a Kalman filter.

 
 
 Kernel methods have been used in other aspects of localization, such as an SVM for ranging error estimation based on CSI in an UWB system in [ 224 ] or SVM for pose recognition in [ 110 ] , which is then used to match for the appropriate RPs. Indoor vs outdoor classification was done in [ 187 ] before location estimation using particle filters. To improve the robustness of the localization process against noise, [ 182 ] proposes a solution for AP selection and classification, then uses SVR to reconstruct the RSSI values of the non-selected APs.

 
 
 The choice of the kernel has an important impact on the solution. Ref. [ 324 ] suggests using kernel canonical correlation analysis to better capture the correlation in the signal (RSSI) space and the coordinates before the localization, where Gaussian and Matern kernels are used for signal space and physical space, respectively. In [ 325 ] the authors propose using a sum-of-exponentials kernel that is tolerant to missing values and sensitive to feature differences in RSSI based cellular positioning. Ref. [ 160 ] uses a kernel that incorporates the spatial structure of the training set; it also proposes a kernel that utilizes the possible correlation between the dimensions in the coordinates. In [ 108 ] a hybrid kernel consists of a local kernel (an RBF kernel) and a global kernel (a polynomial kernel); jointly they take into account the impact of nearby and distant RPs. In [ 80 ] an Import Vector Machine (IVM) was used for NLOS classification for ToA based ranging in UWB systems; the authors point out that ISM has less complexity and represents better classification probability.

 
 
 Gaussian Process Based Methods: 
There are several other ML-solutions based on Gaussian Process (GP) [ 218 , 326 , 175 , 113 , 132 , 94 , 244 ] . GP can be viewed as Bayesian alternative to the kernel methods [ 23 , 327 ] , where the goal is to infer the posterior distribution of the labeling function for given observations. In the training phase, we seek good values for the mean and the covariance of the GP. The latter is usually captured with kernel functions, such as RBF or Matern.

 
 
 In [ 326 ] the authors propose to use GP regression on the training data to build a continuous distribution of the RSSI for each AP. For localization, for given RSS observations, they apply a Maximum Likelihood Estimator algorithm to infer the coordinates of the target. To capture the correlation between different locations they studied three different kernel based correlation functions: Gaussian, Matern, and Quadratic kernels. Ref. [ 328 ] uses a GP to model the probability distribution of the RSSI values, then for a given RSSI observation, and using Bayes rule, the location is estimated by a weighted combination of the RPs’ locations. Ref. [ 329 ] describes a method for dynamically estimating and calibrating the RSSI radio map using GP; for this the standard deviation of the trained GP model is used to measure the accuracy of the estimated position; the final location is estimated using WKNN. We here point out that there are number of other works that use GP (and other ML techniques) to capture the RSSI correlation and distribution [ 330 ] , which can be used for different purposes in wireless systems.

 
 
 For a distributed massive-MIMO system, [ 113 ] introduce numerical approximation GP methods that result in a test Root Mean Square Error (RMSE) very close to the Cramér–Rao bound, which is verified with simulated data. For a UWB system, [ 244 ] uses a number of channel parameters that are extracted from the PDP, such as TOA, RSS, and RMS delay spread for ranging. The first step consists of using a kernel PCA, in which the selected channel parameters are projected onto a nonlinear orthogonal high-dimensional space; a subset of these projections is then used as an input for GPR to provide a ranging estimate.

 
 
 Trees and Ensemble Methods 
Decision trees derive their classification rules by splitting the observation-labels space. Decision trees have been used in a number of localization problems, e.g., for LOS identification [ 157 ] , coordinate prediction [ 290 ] , and localization after AP selection and clustering [ 309 ] . However, many of the papers use them for simple classification problems, or end up using many decision trees in their solutions.

 
 
 Ensemble methods use multiple learning solutions to obtain the final decisions; they have been shown to provide excellent performance even when built as an ensemble of basic ML solutions, e.g., decision trees.
Random Forest (RaF) is one such ensemble learning method; it incorporates multiple decision trees. It has been used frequently in localization. For example, [ 204 ] applies Volume Cross-Correlation on the CSI to acquire the TDOA, then uses a RaF for classification-based localization using multi-path information. The proposed solution uses a combination of both ray-tracing and measurements to enhance the localization performance. Ref. [ 53 ] uses the RSSI in WiFi networks to provide the location in terms of both coordinates and room-level prediction. In the offline phase, fingerprint data are pre-processed, ensemble classifiers (based on GMM) are trained for room prediction, and a RaF regressor is trained for location prediction. In the online phase, data is pre-processed, soft cluster membership is determined, and room and location are predicted. Ref. [ 79 ] utilizes a multi-antenna system to build fingerprints with different features, such as RSSI, power spectral density, and other statistical features. For each feature a RaF is trained as classifier. To increases the robustness of the location estimates, the solution uses multiple samples and classifiers, where an entropy metric is used to choose a robust classifier and a stable time instant. Then the location is the mode of the location predictions constrained to be within the union of the predicted locations from the selected classifier and at the selected time instant.

 
 
 Other ensemble learning methods have been used as well, such as gradient boosting regression forest (GBRF) [ 142 ] , and AdaBoost [ 96 , 92 , 57 ] . Due to fluctuation of the RSSI signals, the RSSI distance might not reflect the true location distance; to address this [ 142 ] proposes a ﬁngerprinting method by transforming raw RSSI into features with a learned non-linear mapping function using GBRF. The idea is to pair the RPs in positive and negative pairs (based on a predefined threshold), then create a loss function that ensures that the similarity is preserved. The mapping function (here GBRF) is then trained. Finally, the localization is performed using WKNN with the new mapping function. Ref. [ 57 ] employs Adaboost for passive localization, where the phase information in CSI is used to construct the ﬁngerprint map. Through continuous iteration with the Adaboost algorithm, the sample weights of the training sets are continuously adjusted to prepare for classiﬁcation. The final location estimate is weighted average of the locations of the top four predicted RPs.

 
 
 Finally, we point out that a number of the above-discussed papers compare the performance of their solution against other ML algorithms. Furthermore, a number of works studies the performance of different ML solutions, such as Ref. [ 94 ] for indoor localization using cellular signals. Ref. [ 237 ] compares KNN, RaF and SVM for device-free localization using CSI. Ref. [ 129 ] suggests localization using frequency modulation and digital video broadcasting terrestrial signals, comparing KNN and, SVM and SVM with Ensemble Learning.

 
 
 

#### V-A 2 Deep Learning Solution 

 
 Nowadays, NNs are among the most popular ML architectures. This can be partly attributed to the fact that many of the successful DL architectures are based on them. Thus we start this section with a review of the localization solutions based on Feedforward NNs.
 Feed-Forward NNs: 
Due to a number of attractive properties of NNs (see Sec. II-A ), many NN based localization solutions have been proposed; in the following we highlight representative samples.

 
 
 Ref. [ 247 ] uses the RSRP of the three strongest BSs along with their locations in an LTE network as input to localize the user. Ref. [ 192 ] uses an NN, with three hidden layers, that takes RSSI from a number of BSs. Data augmentation techniques, e.g., masking some of the BS values, improve the robustness of the solution. In [ 250 ] , two networks are trained for two different features, namely TOA and RSSI from a WiFi systems, and the location is a weighted combination of the outputs of the two networks. Ref. [ 151 ] uses a hierarchical localization solution, where for each indoor location, an NN is trained. During the online phase, the real-time RSSI measurements (from WiFi or/and cellular) are first used to identify the environment, and then they are passed as input to the corresponding trained NN.

 
 
 Using directional antennas in a WiFi system, [ 206 ] builds fingerprints using RSSI (or CSI amplitude) and uses an NN with two hidden layers as a classifier to predict the closest fingerprint to the observed values. Using a rough estimate of the AoA provides an improved localization accuracy. For an 8 × 2 8\times 2 MIMO system, [ 238 ] uses CSI amplitude at 16 antennas and 924 sub-carriers as input to several NNs with different hyperparameters (i.e., an ensemble learning solution); the final location is a combination of the output from all networks. This paper also studied different combining techniques; a method that uses both the median and weighted average of the location estimates provided the best result.

 
 
 The above architectures usually require the backpropagation algorithm for training, which is generally time-consuming, Extreme Learning Machine (ELM) is a simplified feed-forward NN architecture with one hidden layer. The weights of the first layer are set to random values, while the weights for the hidden layer are usually calculated with the least square fit. A number of works try to utilize the simplicity of such architecture in localization, [ 159 , 216 , 150 , 173 ] . In [ 159 ] , during the offline phase, a WiFi RSSI fingerprint database is created, and an ELM is trained on the data. During the online phase, fingerprints are still collected at some known locations, and the solution is updated according to the new fingerprints too. Then the latest ELM is used to predict the coordinates. In Ref. [ 173 ] , during the offline phase, the data is first clustered using k-means clustering, an ELM is used to classify the data to one of the clusters, and a dedicated ELM for each cluster is trained. During the online phase, the RSSI values are first classified and then the ELM of the associated cluster is used.

 
 
 A framework that utilizes features from different technologies for target tracking was introduced in [ 111 ] . The inputs to the NN are: the RSSI measurements from WiFi, Bluetooth and XBee (a wireless connectivity module used for IoT) from different nodes, plus yaw readings when available. The yaw readings can be extracted by filtering measurements of some wearable sensors (accelerometer, magnetometer and gyroscope). The probabilities of fingerprints are passed through a Gaussian Outliers Filtering process, and the output can then be weighted and combined to provide the location estimate, which in turn is fed to a particle filter for target tracking.

 
 
 For MIMO-OFDM systems, [ 297 ] proposes a fingerprint-based localization scheme (which is called DeepFi) that utilizes the magnitude of the CSI at 90 sub-carriers from three antennas. It uses the weights of NNs with four hidden layers to represent the fingerprints. Training the NNs is done by a greedy learning algorithm using a stack of RBMs (see sec. II-A ). After the pre-training and supervised fine tuning, the output of the NN is a reconstruction of the input data. 11 11 
 11 
 
 
 
 Note that this can be as well viewed as an Autoencoder structure, which we discuss below. During the online phase, using a number of RSSI realizations (packets), a Gaussian RBF kernel is used to represent the likelihood probability of the observed data when i th i^{\rm th} location (RP) is true, from which Bayes rule can be used to calculate the posterior probability of location i i . The final location is the weighted average of all RP locations. Ref. [ 331 ] proposes a scheme (called PhaseFi) that utilizes the phase value to construct a fingerprint database, where a linear transformation of the phase value as a calibration step, improves the stability of the phase values. The weights of a three-hidden-layer-NNs serve as fingerprints.

 
 
 NNs in passive localization have been considered in [ 58 ] , where amplitude and the calibrated phase of CSI are used as a hybrid complex input feature to the NN to detect the presence of a human. In mmWave communication systems, [ 183 ] integrates NNs in a cooperative WLS estimator. NNs were used for ranging error mitigation in [ 201 ] , where the authors use RSSI values as input to an NN, which then predicts the ranging error; a localization algorithm such as least squares can then use the adjusted range values.

 
 
 Convolution NN (CNN): 
Recently, several works proposed CNN based localization schemes, since those have shown good performance in computer vision, as discussed in Sec. II-A .

 
 
 Since the number of observed RSSI values in urban areas is large, several works reshape the observed RSSI array into 2D or 3D images and use it as input to a CNN network. Ref. [ 281 ] uses a CNN, applied to the RSSI image, to classify the location in one of several buildings and floor levels. Modifications of the RSSI images were proposed in [ 316 , 105 ] : Ref. [ 316 ] augments the features with correlation values (between the RSSI from the AP at all RPs and their locations). The augmented image is used as input to a hierarchical localization structure, where initially a CNN predicts the floor number, then a CNN associated with the chosen floor predicts the corridor number, and finally a CNN associated with that corridor is used to estimate the coordinates. Ref. [ 105 ] proposes hybrid RSSI features that contain the ratio of the contribution of the RSSI value to the fingerprint. Viewing the RSSI values as time series, Ref. [ 307 ] uses CWT (see Sec. IV ) to produce a 2D time-frequency image; the CNN predicts the closest reference points to the target, and KNN is then used to infer the coordinates of the target. In another hierarchical localization model, [ 256 ] uses RSSI values from all APs over different time instances to form RSSI-time 2D images to predict the building, floor and coordinates. BLE RSSI observations over time are used in [ 286 ] for vehicle and pedestrian detection/localization in a smart parking system. In [ 227 ] , the RSSI values from several APs are used to predict the location with two capsule networks, one for the RSSI values in 2.4 2.4 GHz and one for RSSI values in 5 5 GHz. Capsule networks are CNN architectures with added capsule modules to track the hierarchy of the objects in the images [ 332 ] .

 
 
 Figure 5: The structure used in [ 333 ] , with three CNN layers and two fully connected layers. The input features are the amplitude of the CSI at 30 30 sub-carriers and 30 30 time instant, these values are captured from three antennas to constitute an image with three channels. They use 10 10 kernels at the first layer. Since the input image is relatively small and to use deep structure, the authors propose to maintain the size of the image for the two consequent CNN layers (through padding and stride equal to one), the output is probability of the RP. 
 
 
 With the improved efficiency of NNs through CNNs, it is possible to utilize high dimensional features. The ”ConFi” scheme [ 333 ] is a multi-layer CNN network that uses amplitudes of the CFR at the available sub-carriers (in a WiFi system) at a number of time instances to form an image; realizations at different antennas can be used at different CNN channels (see Sec. II-A ), and the output is the probability that a target is located at a given RP, see Fig. 5 . The final location is estimated by averaging the location of the RPs with their predicted probabilities. The so-called ”CiFi” scheme [ 205 ] uses coarse but stable estimates of the AoA, based on the pair-wise phase difference between antennas. The provided example uses the 30 30 available sub-carriers and 960 960 time instances to form 16 16 60 × 60 60\times 60 images for training. During the online phase it predicts the location by fusing the location of the most probable reference points. Ref. [ 242 ] uses the learned spatio-temporal features through dual stream 3D CNNs to approximate the posterior distribution of the mobile device’s location with a GMM. It creates two 3D images of the calibrated phase and amplitude of the CSI, where the height, the width, and the depth (number of channels) are, respectively, the amplitude/phase from different packets, different sub-carriers (30 available) and Tx-Rx antenna combinations. In a three-antenna system, [ 232 ] creates three-channel images that have the amplitudes of all sub-carrier × \times packet samples in one channel, and pair-wise phase differences between the antenna signals in the two other channels. They use ShuffleNet, a computationally efficient deep CNN architecture, to predict the most probable RPs.

 
 
 For massive MIMO systems, [ 212 , 215 ] uses images in the angle-delay domain. For a Uniform Planar Array, [ 212 ] proposes 3D images that have power values in horizontal, vertical, and delay domains. The images are fed to a CNN network that uses the inception module and different kernel sizes (due to different sparsity at different domains). The inception module has been used in famous Deep NNs architectures such as AlexNet, where the output of different kernels are concatenated, which allows better feature extraction.

 
 
 CNNs have been used for device-free localization in [ 271 , 233 , 54 , 59 ] . For instance, [ 271 ] creates images using the time sequence of RSSI values and their CWT; the solution then detects the presence of users indoor. In [ 233 ] , the authors use 1D CNN networks with either RSSI values from all antennas over several packets or CSI amplitudes from all sub-carries and antennas. They notice the superior performance of CSI based solutions to detect the region of the target. For ranging in UWB systems, Ref. [ 225 ] uses a CNN network for ranging error mitigation, proposing two approaches,: one for NLOS detection and another one to estimate the ranging error. In an industrial environment, where ranging based on ToA is difficult, due to the increased multi-path propagation, Ref. [ 240 ] proposes to use time-calibrated complex CIRs to create images. The authors modified several of popular deep networks such as AlexNet and GoogLeNet. They found that GoogLeNet gives a good complexity-to-performance trade-off. They also propose a distributed framework to reduce the overhead.

 
 
 Recurrent NN (RNN) Based Architectures: 
Several recent works utilized sequential RSSI values for localization. Ref. [ 334 ] uses a sequence of RSSI values as input to cascaded RNNs to estimate both the building and the floor numbers. In a system with multiple BLE anchors, [ 284 ] proposes efficient construction of the fingerprint database for real-time systems with RSSI samples, where LSTM is used for localization. In a cellular system, Refs. [ 115 , 283 ] use the RSSI history from only the associated cell tower to track users’ location with an LSTM, where the inputs are (cell tower ID, RSS) sequential pairs and the output is the 2D coordinates. To train the LSTM, Ref. [ 283 ] proposes using KNN and HMM to generate synthetic measurements based on the previously observed ones. In [ 272 ] , the authors apply a sliding window to the sequence of RSSI samples; in each window they calculate five features (minimum, maximum, and three quartile values) from each AP. They then feed them as a sequence of vectors to an LSTM, see Fig. 6 for more details.

 
 
 Figure 6: The structure used in [ 272 ] : Several consecutive RSSI samples from each AP are collected ( n n samples and r r APs), for a given time window (of size w w ) and for every AP, 5 5 statistical features are computed: (minimum and maximum RSSI values, and the first, the second and the third quartiles), this results in 5 ​ r 5r local features and a sequence of length n / w n/w . The resultant sequence is fed to two LSTM layers to extract high level features, which are then fed to fully-connected layers and regression. Dropout layers combat over-fitting. 
 
 
 In cellular and WiFi systems, [ 149 ] uses RSSI values from cellular and WiFi systems, the solution takes the RSSI fingerprint from multiple APs/BSs at multiple timesteps and passes them to a single-layer RNN which then gives the coordinates. In systems with Unmanned Aerial Vehicle (UAV)-BSs, [ 167 ] proposes using RSSI from three UAV-BSs and nearby WiFi APs to predict the location. The RSSI features are first reduced to 30 features using PCA, and then combined with the rest of the features. The new feature vector is then passed as input to the GRU Network for location estimation. In a related setup, Ref. [ 335 ] uses RSSI values from the reachable WiFi APs and three temporarily deployed UAV-BSs to build a radio-map, and uses LDA to select the subset of APs to reduce computation time. It then feeds the RSS sequences to an LSTM to predict the location. In an MIMO-OFDM system, [ 133 ] utilizes the CSI amplitudes at different antennas and sub-carriers for localization. After prepossessing, shifting and polynomial regression (for smoothing) of the CSI, the correlation matrix, CSI amplitude and SNR are used as candidate features for several ML solutions. The authors point out that LSTM outperforms other solutions when feeding the data sequentially data (i.e., utilizing user’s trajectory).

 
 
 Composites of CNNs and RNNs have been considered in [ 191 , 255 ] . Ref. [ 191 ] uses the consecutive MR samples in a cellular system to generate a smooth trajectory
consisting of predicted locations. It first divides the entire region into cells, then creates images with height and width similar to the grid structure and fills it with features from the MRs. Next, it feeds these images to a CNN to estimate a score for every potential location based on the extracted spatial features. Then it uses the windowed scores as input to a multi-layer LSTM and finally with regression to generate the trajectory. Note the role of the CNN here is to learn local spatial features from each individual MR. Ref.
 [ 255 ] uses sequential RSSI values from nearby APs for localization. The data are fed to a 1D-CNN that captures the features. Then a GRU is used to capture the time dependency. The output of the RNN is fed to a Mixture Density Network (MDN) that learns the conditional probability distributions of the locations (see sec. II-A for mixture models).

 
 
 A somewhat different approach is taken by
”DeepTAL” [ 202 ] , which handles the TDOA measurement error or missing data in an asynchronous system, where the system predicts the target state (moving /static) and outputs the TDOA. The TDOA and the difference of the TDOA are used as the training input. The location is then solved with a quantum-behaved particle swarm optimization algorithm.

 
 
 Auto Encoders: 
In the localization literature, AEs have been used heavily. Although AEs are usually unsupervised learning techniques, in the localization literature, they have been used in conjunction with many of the earlier supervised solutions. One popular approach is to use the AE to extract robust feature representations, where the output of the encoder will be the input of a localization module. Ref. [ 103 ] integrates AE into ELM based classification localization, where the AE is used to extract high level features, and the output of the AE’s encoder is used to replace the random projection of an ELM. As discussed in later sections, it is relatively easy to integrate different regularization approaches when training the ELM, which in this case depend on the encoder output. A hierarchical tuning mechanism is used to train the solution, first tuning the classifier parameter, and then adjusting the encoder parameters. Ref. [ 336 ] uses the output of the Stacked AE (SAE) with a one-dimensional CNN to provide the location based on the floor/building and the target coordinates. Similarly, [ 74 ] proposes a multi-output network that uses the encoded features to predict the building, the floor, and the coordinate. An SAE followed by two FC layers and then an argmax to predict the multi-label classification is used in [ 291 ] . Ref. [ 305 ] uses the AE to provide a low dimensional representation of the RSSI and magnetic field signals. AEs can also be applied to device-free localization [ 62 , 295 , 56 ] . Convolutional AE (CAE) uses the output of the encoder as input to a classifier that predicts the RP where the target may be [ 62 ] . The input to the CAE is an image constructed with the difference between the (target free) RSSI values and the online RSSI values; Fig. 7 elaborates on the method. Ref. [ 295 ] estimates the target’s location, activity, and gesture, based on the RSS measured signals. The approach first denoises the signal with a four-level wavelet decomposition, then uses Sparse-AE to project the high dimensional data to low-dimensional features and finally feeds the learned features into the softmax classification and regression model. The AE could also be used for data transformation: the device heterogeneity problem, can be tackled, e.g., by an AE that maps the features observed by a test device to features that corresponds to the device used for acquiring the database [ 280 ] .

 
 
 Figure 7: The system proposed in [ 62 ] formulates the device-free localization as a classification problem to predict the location of the target in one of L L RPs, where the area is divided into L L grids. The proposed network is based on a Convolutional network that is pre-trained to extract useful features as an encoder of a Convolutional AE architecture. The encoder is then attached to fully connected layers and softmax. The input image is constructed using the RSSI measurements from D D APs that act as a TX and an RX in turn. They collect the RSSI between all the APs when (i) no target is present in the area, (ii) when a target is present. The RSSI images are the difference between the two cases. 
 
 
 In an alternative approach, the AE could be used in fingerprint-based localization to assess the similarity between the observed features and the collected fingerprints. In ”BiLoc,” a fingerprint localization solution uses AoA estimates and the amplitude averaged over two antennas as input to AEs, and then employs the weights of the trained AEs as fingerprints [ 331 ] . The online phase uses the AEs to reconstruct the observed features; the level of similarities (calculated with RBF) will define the probability of the target being in a specific location. The final target location is the average of the weighted location of the fingerprints. Ref. [ 298 ] uses the location along with the latent variable of the AE to reconstruct the observed signal. The location is based on the most similar RP. Ref. [ 270 ] builds SAEs corresponding to each fingerprint. In the online phase, it tries to reconstruct the observed features and compare the similarity using RBF to provide a probabilistic location estimate.

 
 
 In another approach, the AEs are used to provide a pre-training method before complete training to fine-tune the parameter. As highlighted in Sec. II-A , the pre-training technique is not limited to AEs; in fact, one of the early works uses RBMs [ 337 ] . For pre-training, not all the data need to be labeled. These methods have been used in [ 259 , 260 ] , while [ 260 ] uses a DNN to provide coarse location estimate and an HMM to refine results. In [ 259 ] the solution is composed of a NN, CNN, and followed by a probabilistic
technique for fusing the candidate location estimate.

 
 
 
 

### V-B Semi-Supervised learning 

 
 As introduced in Sec. II-A , in semi-supervised learning, the solution uses a combination of limited (but important) labeled data and a large subset of unlabeled data. Similar to supervised learning, assumptions such as smoothness are required to allow generalization. With high dimensional data it could be difficult to assess the similarity between points [ 338 ] , and the manifold assumption could offer a remedy for this. In fact many of the semi-supervised learning solutions utilize this assumption. The taxonomy of this section is different from Sec. V-A as the approaches are usually different.

 
 

#### V-B 1 Manifold Learning 

 
 Manifold alignment is a popular ML framework. It allows a transfer of information between datasets under the condition that there is an underlying common manifold [ 339 ] . There are many applications for this, such as dimensionality reduction, and data visualization, see Sec. II-A . Graphs are sometimes used to approximate the point relations on the manifolds [ 338 ] . Different manifold alignment algorithms are usually different in how neighborhood graphs and the corresponding distance between nodes are established. Manifold alignment has been used in the localization literature. In semi-supervised learning, the knowledge about the ”distance” between the observed data can be used to build the radio map. Works including [ 267 , 274 ] use the Laplacian Eigenmap manifold alignment approach. In this technique, a weighted graph between the data points (including labeled and unlabeled data) is constructed to preserve the local geometry of the data. Then the mapping uses the eigenvectors of the graph Laplacian. For example, labeled RSSI data from nearby APs as well as unlabelled timestamped traces can be used to construct the graphs (which incorporates the physical relations of labeled fingerprints), using the labeled data as hubs to construct the graphs [ 274 ] . Ref. [ 261 ] also uses the RSS and RSS traces (based on crowdsourcing), which can integrate the spatial correlation property into the graph. In Ref. [ 265 ] the authors utilize ELM for deep feature extraction, which they use along with the graph Laplacian to define the objective of the semi-supervised classification (localization) problem; the weights on the graph are calculated with Gaussian kernels between the RSSI vectors. Ref. [ 340 ] proposes modification of the objective functions to preserve wireless propagation model based estimated distances.

 
 
 Ref. [ 341 ] uses manifold regularization [ 342 ] to develop a semi-supervised RSSI localization solution, which first produces pseudo-labels by optimizing a time-series graph Laplacian SVM; the pseudo-labels are then integrated into a learning framework. It combines the manifold regularization into a transductive SVM 12 12 
 12 
 
 
 
 Transductive SVM belongs to the set of transductive learning, where the goal is not to learn the general mapping function, but rather it is restricted to the given test data [ 343 ] . to balance the contribution of labels and the pseudo-labels [ 343 ] . In [ 344 ] the authors use ELM for localization. The solution for ELM parameters takes into account the manifold regularization based on Laplacian graphs in WiFi. Ref. [ 345 ] adds the RSSI measurements from BLE; since WiFi and BLE signals exhibit different propagation conditions (different transceivers capabilities and transmission powers) two graph Laplacians are constructed to capture the smoothness in WiFi and BLE.

 
 
 A Siamese network architecture uses two identical NNs to compare two inputs; it was used in [ 239 ] as supervised or semi-supervised CSI-based localization solutions , see Fig. 8 . Stemming from the fact that location influences the values of large scale parameters, the idea is to use two identical feedforward NNs to map the input features to lower dimensional representations (e.g., location). When labeled data is available the loss function should include the ground truth CSI to coordinate mapping while preserving the feature distances.

 
 
 Figure 8: The proposed Siamese network in [ 239 ] . The network uses two identical NNs f θ f_{\theta} to map high dimensional input features (derived from CSI) 𝒙 n \boldsymbol{x}_{n} and 𝒙 m \boldsymbol{x}_{m} to lower dimensional representations 𝒚 n \boldsymbol{y}_{n} and 𝒚 m \boldsymbol{y}_{m} . The goal of the network is to preserve the distance between 𝒙 n \boldsymbol{x}_{n} and 𝒙 m \boldsymbol{x}_{m} , and 𝒚 n \boldsymbol{y}_{n} and 𝒚 m \boldsymbol{y}_{m} ( 𝒅 m , n \boldsymbol{d}_{m,n} ). The parameter α \alpha is a scaling value used in a semi-supervised learning scenario for distance scale matching [ 239 ] . 
 
 
 

#### V-B 2 Generative and Statistical Models

 
 Ref. [ 263 ] proposes a hybrid generative/discriminative classification and regression learning algorithm, employing a naive Bayes and EM algorithm to construct the generative model. It uses the naive Bayes method to learn the initial probabilistic model parameters from a limited number of labeled samples. The EM algorithm is then employed to gradually improve the parameters using the unlabeled samples. The least-square SVM (LS-SVM) is then trained on the labeled data to perform discriminative learning.

 
 
 Motivated by the success of semi-supervised learning with deep generative models [ 346 ] , Refs. [ 347 , 147 ] use Variational AE based semi-supervised localization that can utilize both labeled and unlabeled RSSI observations. It consists of two components: a latent-feature discriminative model M1 and a generative semi-supervised model M2 [ 346 ] . Both rely on variational inference, which aims at replacing a complex unknown distribution p p with a simpler and more tractable distribution q q (usually Gaussian). The similarity of p p and q q (captured by the KL-divergence) is maximized by optimizing the ”variational lower bound”. In M1 the encoder learns the latent representation of the observation (RSSI in [ 347 ] ), in M2 it uses both the observation and the labels (if present) to learn the mapping. Combining M1 and M2 provides a powerful deep generative model that uses all the available data for location inference [ 347 ] .

 
 
 

#### V-B 3 Other Methods

 
 In practical setups, the environment changes over time, which makes the collected data and the trained model outdated after some time. The goal of [ 211 ] is to keep the trained classification solution of a device-free localization up to date. It proposes a method to label unlabeled data and then use them when drift occurs to re-train the model. Thus the solution automatically re-trains when the uncertainty level rises significantly. The uncertainty is based on KL divergence, which is used as a distance metric to track substantial changes in the distribution of the features. More solutions for the same problem are presented in Sec. V-D .

 
 
 Ref. [ 264 ] proposes a label propagation (LP) algorithm in classification-based localization. By representing labeled and unlabeled data as vertices in a connected graph, the algorithm iteratively propagates labels to unlabeled vertices through weighted edges; the weights of the edges depend on the distance of the observed signal strength values. It can then infer the labels of unlabeled data after this propagation process converges. Throughout the iterative process, the true labels of the collected training data are maintained. A similar LP is carried out for the new observations.

 
 
 In a different research direction, Ref. [ 114 ] uses semi-supervised learning for trajectory learning in a map-less setup. The overall localization solution is a combination of GP for location likelihood learning (probability of RSSI given the location) and a particle filter. The learned trajectory is used to identify the prior of a particle filter. The semi-supervised part of the solution is a weighted version of unsupervised and supervised dimensionality reduction techniques, PCA and LDA respectively, which is employed to learn the landmarks (e.g., a room) for the given RSSI signals. Once identified, the landmarks are used to identify the trajectories’ starting and end points.

 
 
 
 

### V-C Unsupervised learning 

 
 As discussed in the previous subsection, unlabeled data can still be useful as they contain the structure and distribution of the features. For this reason, many solutions pre-train the model with unlabeled data. This step can be used in combination with any other ML solution. In V-A , we discuss several algorithms that use SAE to pre-train deep networks. Other architectures have been used as well, such as RBM and Deep Belief Networks (DBN). For instance, [ 348 ] uses DBN for deep feature learning from RSS measurements (after noise reduction); the features are then fed to another ML solution (classification/regression) for location estimation. Unsupervised learning was used to enhance localization solutions, e.g., for AP selection (e.g., [ 309 ] using k-means clustering), data filtering (e.g., clustering erroneous crowdsourcing data for outlier detection in [ 349 ] ), and training-device to testing-device mapping (e.g., by introducing an online step to train linear mapping between the two devices based on coarse labels [ 320 ] ). However, in this section we focus on solutions that use mainly unlabeled data for localization.

 
 
 In addition to pre-training, unlabeled data could be used for radio-map construction. The ”WILL” scheme [ 279 ] for room number prediction creates a logical floor plan using RSSI traces and accelerometer readings (to identify mobility) from any user within the service area. The plan construction is based on clustering techniques on RSSI stacking difference (RSSI reading difference from one AP to all other APs); an example of one of the used clustering is k-means clustering. Each virtual room on the logical map is assigned a representative fingerprint for the localization phase. The paper also provides a matching algorithm to map the logical plan to the ground-truth floor plan that is usually available to real-estate administrators. Ref. [ 278 ] proposes a framework for unsupervised localization that uses readings from accelerometer, compass, gyroscope, as well as WiFi readings the users report while moving naturally inside a building. It first tries to identify some “seed landmarks”, which are certain structures in the building, such as stairs, elevators, entrances, that force the users to behave in predictable ways. It then dead-reckons the devices starting from a known reference location. Since the sensors might show some consistent measurements in the indoor areas, e.g., dead-spots or sudden increase of reflections, the algorithm employs K-Means clustering to extract unique sensor signatures that can increase the localization accuracy. Note that the framework relies on one-time global truth information, e.g., the location of a door, or staircase, or elevator. Ref. [ 277 ] aims to fit the RSSI traces into the structure of the environment without use of labels. Localization is obtained from use of HMM and global-local optimization that considers the solutions that do not violate the signal propagation to restrict the search space.

 
 
 In a different set of approaches, [ 302 , 194 ] devise customized loss functions that can be used to train the localization model without the labels. The proposed solution in [ 194 ] can run in a semi-supervised or in an unsupervised way. It assumes that the user can estimate the distance to the APs with RSSI measurement (through path-loss model, multinomial fit or NN), or using an FTM protocol. The paper proposes several cost functions that can be used to assess the ranging accuracy of the APs, e.g., the distance between the predicted location and ranging distance (for unsupervised) or difference between the predicted location and true location for the few labeled points (for semi-supervised). Finally, the gradient of these cost functions can be used to train the models. Ref. [ 302 ] uses the unlabeled data with the customized cost function to predict the location of a subset of APs with known locations.

 
 
 Refs. [ 193 , 85 ] propose graphical model based solution (see Sec. II-A ). The location (points on a grid) (in both works) along with the transmitted discrete power levels (in [ 193 ] ) are modeled as latent variables. Assuming the RSSI values to be approximately normally distributed and to be independent from one APs to another, the solution uses GMM to model the probability of the received RSSI. The latent parameters are then estimated using EM algorithm. To solve the identifiability problem (to match the predicted grid pint to the true one), in [ 193 ] the knowledge of the APs locations along with simple pathloss model is used to initialize the EM algorithm.

 
 
 Since a number of solutions use fusion to predict the final location of the target, [ 161 ] proposes an unsupervised learning method to improve the fusion, using an extended candidate location set (concatenation of the top classifiers’ outputs), and designing a joint estimate of weights and location under an unsupervised optimization framework. Reliable predictions are assigned higher weights than the unreliable ones, so the true location of the user should be close to reliable predictions.

 
 
 Finally, Multidimensional Scaling (MDS), a dimensionality reduction technique, has been used extensively for WSN localization [ 21 , 350 ] . Similar to LLE (see Sec. II-A ), MDS performs projection to a lower dimensional space while preserving the known inter-wireless nodes distances; the projection results in the node configuration in space. The distance can be estimated using wireless signals, e.g., RSSI or ToA readings; this has been recently utilized as a cooperative localization technique for RFID (and in general IoT) nodes [ 351 ] , and cognitive radio [ 352 ] . We here limit the discussion of these methods as wireless signals are usually used to only calculate the inter-node distances; we refer the interested reader to other recent survey in this subject [ 352 ] .

 
 
 

### V-D Transfer Learning 

 
 Transfer learning (TL) is used in ML to exploit the knowledge acquired in the source domain for the learning task in the target domain [ 353 ] . There might be several reasons to use TL: When the acquisition of the data that matches the specific application domain could be difficult, there might be abundant data in one domain, but limited data in the target domain. In localization, the system might be configured or set up in one environment and deployed in a different environment, e.g., in dynamic systems. Alternatively, the data could be available in both domains, but retraining the model is difficult. TL could be used to arrive at a good solution in the target domain without retraining the solution from scratch.

 
 
 Some of the early works to apply TL in localization using RSSI are summarized in [ 354 ] . TL over space (e.g., different parts of the building), devices and time, is considered, where in each of these problems the distribution of data could be different over time, space or target devices. For TL over space, TL is formulated as two optimization problems, where in the first the underlying semantic manifold of the signal is extracted, which can be used as constraints for the second one, where the unlabeled data in the target domain are labeled. For TL across devices, the problem can be formulated as multi-task learning. For a TL over time, [ 355 ] proposes a solution based on modified manifold regularization; this is motivated by the fact the distributions of the data (RSSI), while different over different time instance, are expected to be similar in low-dimensional space (the physical space). The solution uses labeled fingerprints and a few unlabeled data in the offline phase, which can be used to learn the localization function. During the online phase, it collects a few measurements on some of the fingerprints and additional unlabeled data points. With the modified manifold regularization it learns the joint mapping function for localization to update the mapping function learned during the offline phase.

 
 
 Manifold alignment has been used for TL. Ref. [ 262 ] uses a simulated (e.g., by ray-tracing) radio propagation map as source domain, then use it along with a few calibration fingerprint in the target indoor environment. This study constrains the two domains to have similar spatial correlation of the RSS values, and then used LLE (see Sec. II-A ) to find a low dimensional representation, and find the nearest location to the observed features. Manifold alignment for TL was used in [ 251 ] as well, where the source domain has fingerprint data measured at different times and using different devices. In [ 98 ] , with the source and the target domains, respectively, containing labeled RSSI fingerprints and online RSSI readings, the transfer learning is performed by mapping the source and target domains into a latent feature subspace and maintaining both global and local structural consistency. In the latent subspace, conventional matching or ML algorithm can be used to yield the location estimates.

 
 
 In another approach, [ 356 ] uses the labeled fingerprints (collected offline) with additional online APs readings as source domain, and the online target reading as the target domain. Using the data from both domains a transfer learning kernel is used to learn a domain-invariant kernel, which can be used with SVM for localization. In [ 124 ] , in order to reduce the offline training overhead in the new environment, the TL-based framework consists of two parts: metric learning and metric transfer. The metric learning part learns the distance metrics from source domains by maximizing the statistical dependence between the signal features and the corresponding labels. The metric transfer part identiﬁes the most suitable metric for the target domain by minimizing the data discrepancy between target and source domains. Sometimes the TL solution might not account for the entire environment structure: Ref. [ 123 ] uses fuzzy C-means clustering aimed at minimizing the effect of environmental changes on the surrounding area; based on this the radio map can be reconstructed.

 
 
 TL with DL solutions have been proposed recently. Ref. [ 243 ] uses TL between different antenna setups in a massive MIMO system, where the input to a deep CNN network is time-domain CSI (raw values, amplitude, and phase). Once the network is trained, the lower layers of the network could be retrained with limited data points for the new antenna setup. TL between different environments is considered in [ 177 ] . The authors use NNs to solve the RSSI based localization problem. It is done through pre-training the model in one environment and then performing TL to another one, where the source and target domains are separate propagation environments. The solution is applied to two different floors of one building, which have a similar structure and AP placement. The TL only uses 30 % 30\% of the available data in the target floor (domain).

 
 
 

### V-E Other learning Structure 

 

#### V-E 1 Reinforcement Learning

 
 Reinforcement Learning (RL) frameworks usually have at least one agent that has to take sequential actions in a given environment such that the cumulative reward is maximized. RL is one of the hot research areas in ML. It has also gained considerable attention in wireless communications for dynamic systems, such as resource allocation. Recently a few papers started to explore RL for localization. Ref. [ 285 ] proposes a system in BLE that takes RSS values and previous locations information, and considers the reward to be whether the solution reaches known RSS values or an RP. Ref. [ 357 ] views the localization in a WiFi system as Markov Decision Process (MDP), then used a model-free algorithm (Deep Q-Learning) to find the policy that progressively localizes the target with ”right” mapping from observable states to actions. In the solution, the states include the RSSI values, action history, and the coordinate of the current center. The action space contains five motion directions to move the expected location window with different radius parameters. The reward is a function of the intersection between the window and the ground truth. Ref. [ 51 ] uses RL to schedule the exchange of signals in cooperative localization. The solution views the links as agents, the measurement decisions as actions. They use the distance, covariance values and the number of nodes that did not achieve a localization quality threshold to be the observations.

 
 
 

#### V-E 2 Federated Learning

 
 In federated learning (FL), the training is carried out at the target or distributed computing edges. This collaborative framework has many advantages, such as reduced training load and, more importantly, maintaining users’ privacy, as the user does not need to share private data with the network. FL usually assumes the presence of a centralized entity that updates the model by combining the locally (at the user side) trained models and broadcasts it back. There have been limited research papers that use FL for localization. However, this is expected to change as location information is one of the basic private pieces of information. One recent work [ 273 ] proposes an FL scheme that improves the reliability and the robustness of RSS fingerprint-based localization, while preserving the privacy of the participants, using an NN that predicts the coordinate of the users. After the model is trained at the users’ sides, the central node weighs the NN weights by their number of used samples and broadcasts the updated model back to the users.

 
 
 
 
 

## VI Datasets 

 
 The success of ML is predicated on the availability of suitable datasets. In this section we first summarize the types of datasets that were used in the literature for localization, and then we list the datasets that are currently publicly available.

 
 

### VI-A Used Datasets 

 
 We can distinguish three main types of datasets that were used in the localization literature:

 
 • 
 
 Measurements . Here the authors carry out experiments using real devices; the collected data are used to train and test the proposed models. One of the following two data collection methods is used

 
 – 
 
 Off the shelf hardware . Commercially available devices have been the most prevalent choice, especially when only the RSSI signals are used. For instance, mobile phones can be used to collect the RSSI signal in WiFi or cellular systems, e.g., [ 90 , 279 , 280 , 62 ] . Sometimes the development of special software is needed to extract the desired features. For example, in [ 279 ] the authors developed an application to collect WiFi signal and sensor data. Mobile service providers could use systems logs available in their networks as in [ 187 , 188 ] . A number of datasets with annotated RSSI signals are publicly available; Ujiindoorloc is one of the early datasets [ 358 ] , where annotated RSSI measurements have been recorded in a number of buildings and for a large number of APs.

 
 
 Collecting more advanced features might require more specialized hardware or software. For instance, to collect the raw CSI signal, several papers rely on Intel WiFi link 5300 5300 NIC chipset [ 359 ] . They use modified chipset firmware [ 331 , 300 , 232 , 231 ] , which provides access to the CSI at three antennas. To use the FTM protocol, [ 194 ] uses the Intel AC8260 WiFi chipset that supports FTM functionality [ 360 ] . For UWB, the DecaWave DW1000 transceivers, [ 361 ] , are popular [ 202 , 223 , 193 , 225 ] .
A number of papers introduced modifications to existing localization systems. For instance, the authors in [ 362 ] test the Pseudolite system , where they have to deploy the system indoor. Switched-beam antennas with a USRP board can acquire directional RSSI in WiFi systems [ 206 ] . A testbed with a CC2530 chipset for device-free localization-based Zigbee has been used in [ 295 ] .

 

 – 
 
 Channel sounders are specialized measurement equipment constructed for the precision measurement of wireless propagation channel characteristics [ 36 ] . They have two main advantages: (i) higher accuracy and reproducability of the measurement results, since they are carefully calibrated and use designs that reduce impact of noise, interference, etc.; (ii) since they are custom devices, they can be constructed for scenarios that fall outside the operation range of commercially available systems, such as
carrier frequency, bandwidth, antennas structure, etc.
However, their use is usually limited due to the cost, effort, and the needed expertise to develop them. A few ML-based localization systems used channel sounders, e.g., [ 217 , 238 , 243 , 224 , 220 ] .

 

 
 

 • 
 
 Ray-tracing software. Ray tracing, or more generally deterministic channel modeling, uses 3D models of the environment and the electromagnetic characteristics of the objects in them to provide accurate simulations of the double-directional impulse response. The limitations are mainly due to (i) limitations in the representation of the propagation effects, such as diffuse scattering, and (ii) inaccuracies and limited resolution of the environmental database. Ray tracing is widely used by cellular network operators for network planning, and the accuracy of RSSI prediction is quite high. However, other aspects such as angular spread, are modeled with less accuracy, in particular at higher frequencies. One widely used ray tracing tool for academic investigations is Wireless InSite [ 363 ] , where it was used for localization in, e.g., [ 286 ] .

 

 • 
 
 Statistical Simulations. A number of authors use statistical channel models to generate data to evaluate their solutions. Such statistical models and their limitations are discussed in Sec. II-B .
System evaluations have been based on the 3GPP model [ 213 ] , [ 241 ] , [ 212 ] , the COST 2100 channel model under the 300 MHz parameterization for MIMO channels [ 215 ] , a geometric stochastic channel with parameters based on LTE-Advanced [ 209 , 214 ] , the ITU indoor office channel model [ 364 ] , or LOS and NLOS models based on the Berlin UMa scenario [ 239 ] . A number of these papers have used the model implementations of QuaDRiGa [ 45 ] , a public-domain channel simulator that implements a number of the above-cited as well as other channel models; it allows spatial consistency of the simulated channel, and permits imposition of geometric structure in the considered environment.

 

 
 A number of works use more than one dataset to train and test their model, for instance, [ 217 ] uses ray-tracing data to pre-train the solution before it is fine-tuned and tested on measurement data. Ref. [ 262 ] uses a statistical channel simulator as a source domain in a TL approach. Ref. [ 204 ] uses both measurements and ray-tracing to enhance TDoA based localization.

 
 
 
 
 
 
 
 Measurement 
 | 
 
 
 Simulation 
 | 
 
 
 Ray tracing 
 | 
 
 
 Open Dataset 
 | 

 
 
 
 [ 181 , 67 , 114 , 115 , 52 , 116 , 81 , 82 , 117 , 68 , 119 , 122 , 123 , 124 , 125 , 83 , 84 , 85 , 128 , 129 , 184 , 55 , 130 , 87 , 57 , 131 , 132 , 134 , 89 , 90 , 69 , 135 , 58 , 93 , 94 , 136 , 70 , 95 , 96 , 137 , 138 , 71 , 140 , 97 , 244 , 142 , 143 , 72 , 146 , 98 , 99 , 100 , 101 , 149 , 150 , 249 , 284 , 102 , 251 , 153 , 154 , 252 , 60 , 156 , 157 , 159 , 104 , 160 , 161 , 162 , 163 , 164 , 105 , 185 , 106 , 166 , 167 , 77 , 107 , 64 , 108 , 168 , 169 , 172 , 253 , 288 , 61 , 173 , 62 , 109 , 174 , 175 , 110 , 111 , 254 , 72 , 76 , 65 , 178 , 179 , 49 , 240 , 297 , 228 , 229 , 230 , 231 , 301 , 232 , 282 , 292 , 205 , 233 , 285 , 258 , 234 , 298 , 226 , 202 , 260 , 365 , 235 , 211 ] [ 261 , 262 , 188 , 192 , 263 , 264 , 265 , 266 , 267 , 236 , 269 , 270 , 271 , 272 , 366 , 222 , 302 , 223 , 296 , 193 , 274 , 216 , 217 , 275 , 276 , 277 , 300 , 278 , 279 , 187 , 247 , 280 , 283 ] [ 367 , 368 , 305 , 335 , 191 , 316 , 369 , 238 , 336 , 306 , 243 , 194 , 227 , 286 , 195 , 204 ] 
 | 
 
 
 [ 103 , 79 , 54 , 113 , 182 , 80 , 118 , 120 , 121 , 126 , 63 , 139 , 141 , 144 , 151 , 250 ] [ 155 , 158 , 159 , 164 , 165 , 170 , 176 , 177 , 178 , 294 , 289 , 203 , 213 , 215 , 218 , 219 , 209 , 241 , 364 , 201 , 227 , 239 ] 
 | 
 
 
 [ 152 , 183 , 144 , 258 , 190 , 208 , 204 , 326 ] 
 | 
 
 
 [ 183 , 56 , 132 , 91 , 92 , 53 , 147 , 154 , 156 , 171 , 172 , 74 , 177 , 180 , 255 , 256 , 257 , 268 , 273 , 281 , 370 , 291 , 357 , 336 ] [ 133 , 73 , 75 , 112 ] 
 | 

 

 Table VI: Type of datasets used. 
 
 
 

### VI-B Publicly Available localization Datasets 

 
 Open datasets help researchers to develop solutions without the need for the exhaustive data collection process, and also to measure the effectiveness of the proposed approach against other approaches. One of the best examples of open datasets in DL is the MNIST dataset [ 371 ] , which is a collection of 60,000 60,000 images of handwritten numbers. This dataset is often used by researchers in CV to determine the effectiveness of their algorithms and also by many budding researchers in the field to try out and learn more about CV.

 
 
 For localization, there are number of publicly available datasets. UJIIndoorLoc is one of the most frequently used datasets in Indoor Localization. It consists of RSSI readings from WiFi, collected using about 25 25 different mobile phones at three multi-floor buildings from the Jaume I University, from 520 520 Wireless Access Points. The dataset has 21,049 21,049 data points split across training and validation sets. Another version, the UJIIndoorLoc-Mag was also released, which consisted of sensor readings such as accelerometer, magnetometer and rotation sensor. It consists of 40,159 40,159 measurements.

 
 
 Many open datasets in indoor localization use WiFi RSSI as one of the key features for localization. Generally they are fused with some other attributes such as cellular RSSIs. An examples of WiFi plus Cellular RSSI readings is the PerfLoc dataset which consists of 900 900 data points. This dataset also consists of sensor readings from a large variety of sensors available on an android phone. It was collected in 4 multi-floor buildings.

 
 
 Recently a number of BLE based datasets have been made public, BLE RSS Measurements Dataset for Research on Accurate Indoor Positioning. It has over 4700 4700 fingerprints of BLE RSSI data collected from two zones in the Jaume I University. Android smartphones were used for collecting data from off the shelf Bluetooth beacon systems.

 
 
 
 
 
 
 
 Dataset name 
 | 
 
 
 Released 
 | 
 
 
 Features used 
 | 
 
 
 Wireless technology used 
 | 
 
 
 Location points 
 | 
 
 
 Online availability 
 | 
 
 
 Environment 
 | 

 
 
 
 
 
 Long-Term WiFi Fingerprinting Dataset [ 372 ] 
 | 
 
 
 2017 
 | 
 
 
 RSSI, location coordinates 
 | 
 
 
 WiFi 
 | 
 
 
 212 reference points with 63,504 measurements 
 | 
 
 
 Yes [ 373 ] 
 | 
 
 
 Indoor (Multi-floor) 
 | 

 
 
 
 WiFi Crowdsourced Fingerprinting Dataset [ 374 ] 
 | 
 
 
 2017 
 | 
 
 
 RSSI, local location coordinates 
 | 
 
 
 WiFi 
 | 
 
 
 4648 fingerprints 
 | 
 
 
 Yes [ 375 ] 
 | 
 
 
 Indoor (Multi-floor) 
 | 

 
 
 
 WLAN Indoor Ranging Dataset [ 376 ] 
 | 
 
 
 2018 
 | 
 
 
 WLAN Ranging Signals 
 | 
 
 
 WLAN (IEEE 802.11g/n) 
 | 
 
 
 Approx. 320 files each with a 300x30000 MATLAB Matrix 
 | 
 
 
 Yes [ 377 ] 
 | 
 
 
 Indoor 
 | 

 
 
 
 Rural Sigfox data-set [ 378 ] 
 | 
 
 
 2018 
 | 
 
 
 RSSI, Location coordinates 
 | 
 
 
 Low Power WAN (Sigfox) 
 | 
 
 
 25,638 rows 
 | 
 
 
 Yes [ 379 ] 
 | 
 
 
 Outdoor 
 | 

 
 
 
 Urban Sigfox data-set [ 378 ] 
 | 
 
 
 2018 
 | 
 
 
 RSSI, Location coordinates 
 | 
 
 
 Low Power WAN (LoRaWAN) 
 | 
 
 
 14,378 rows 
 | 
 
 
 Yes [ 379 ] 
 | 
 
 
 Outdoor 
 | 

 
 
 
 Urban LoRaWAN data-set [ 378 ] 
 | 
 
 
 2018 
 | 
 
 
 RSSI, Location coordinates, Low Range Spreading Factor (SF), Horizontal Dilution of Precision (HDOP) 
 | 
 
 
 Low Power WAN (LoRaWAN) 
 | 
 
 
 123,529 rows 
 | 
 
 
 Yes [ 379 ] 
 | 
 
 
 Outdoor 
 | 

 
 
 
 Residential wearable RSSI and accelerometer measurements dataset [ 380 ] 
 | 
 
 
 2018 
 | 
 
 
 RSSI, accelerometer readings, battery level of wearable devices, tagged video of user 
 | 
 
 
 BLE (Bluetooth 4.0) 
 | 
 
 
 42 MB of csv files 
 | 
 
 
 Yes [ 381 ] 
 | 
 
 
 Indoor (4 different residences) 
 | 

 
 
 
 UJIIndoorLoc-Mag [ 382 ] 
 | 
 
 
 2015 
 | 
 
 
 Magentometer readings, Accelerometer readings and Rotation Sensor readings, Location coordinates 
 | 
 
 
 None (Sensor readings only) 
 | 
 
 
 40,159 measurements 
 | 
 
 
 Yes [ 383 , 384 ] 
 | 
 
 
 Indoor 
 | 

 
 
 
 UJIIndoorLoc [ 385 ] 
 | 
 
 
 2014 
 | 
 
 
 RSSI from 520 APs, Location co-ordinates 
 | 
 
 
 WiFi 
 | 
 
 
 933 different locations, 21049 different readings 
 | 
 
 
 Yes [ 383 , 386 ] 
 | 
 
 
 Multi-building, Multi-floor 
 | 

 
 
 
 AmbiLoc [ 387 ] 
 | 
 
 
 2017 
 | 
 
 
 RSSI from FM, TV and GSM signals, coordinates 
 | 
 
 
 FM Radio, TV, GSM 
 | 
 
 
 2697 samples 
 | 
 
 
 Yes [ 388 ] 
 | 
 
 
 Multi-building, Multi-floor 
 | 

 
 
 
 PerfLoc [ 389 ] 
 | 
 
 
 2016 
 | 
 
 
 Motion sensors, RSSI from WiFi and Cellular, GPS 
 | 
 
 
 WiFi, Cellular 
 | 
 
 
 900 points 
 | 
 
 
 Yes 
 | 
 
 
 Multi-building, Multi-floor 
 | 

 
 
 
 KIOS Dataset [ 390 ] 
 | 
 
 
 2013 
 | 
 
 
 RSSI from all available APs 
 | 
 
 
 WiFi 
 | 
 
 
 105 different reference points, 2100 fingerprints 
 | 
 
 
 Yes [ 391 ] 
 | 
 
 
 Indoor 
 | 

 
 
 
 BLE RSS Measurements Dataset [ 392 ] 
 | 
 
 
 2018 
 | 
 
 
 RSS from BLE Beacons, coordinates 
 | 
 
 
 BLE (Bluetooth 4.0) 
 | 
 
 
 4752 different samples 
 | 
 
 
 Yes [ 393 ] 
 | 
 
 
 Indoor (2 university zones) 
 | 

 
 
 
 JUIndoorLoc [ 394 ] 
 | 
 
 
 2019 
 | 
 
 
 RSSI Data 
 | 
 
 
 WiFi 
 | 
 
 
 25,364 samples 
 | 
 
 
 Yes [ 97 ] 
 | 
 
 
 Multi-floor 
 | 

 

 Table VII: Examples of publicly available dataset. 
 
 
 In outdoor localization, examples are the Sigfox and LoRaWAN datasets which consists of three datasets: the rural Sigfox dataset (over 25000 25000 rows), urban Sigfox dataset (over 14000 14000 rows) and the urban LoRaWAN dataset (over 123000 rows). A wide range of sensors and devices were used for sending and receiving messages through these proprietary Low-Power WAN channels in a variety of outdoor settings.

 
 
 Readers are referred to table VII for more information on these datasets and other similar ones.

 
 
 
 

## VII Challenges and Opportunities 

 
 In this section we present a number of challenges for ML-based localization and interesting research directions. While many of them are common to classical solution techniques as well, our emphasis lies on the unique aspects of ML solutions.

 
 

### VII-A Availability of Data 

 

#### VII-A 1 Discussion

 
 One major challenge of ML solutions is to acquire enough data to train and validate the solution. This can be manifested in different ways. When no labeled data are available, supervised and semi-supervised solutions cannot be used. Other solutions still rely on data, such as good representative non-labeled data for unsupervised learning. In fingerprinting solutions, as an example, the fingerprints should be collected when the system is initially deployed, e.g., through a drive-test. However, this is usually labor-intensive; the data must cover the entire service area and be recorded with suitable device(s). Furthermore, to combat the noise, and possible interference, several samples of the fingerprints should be recorded at each sample RP.

 
 
 Environment Drift: 
In realistic systems, the environment usually evolves, e.g., due to installation or relocation of APs or construction of new buildings that act as scatterers. Things become worse in highly dynamic environments, as the recorded data becomes quickly outdated, which degrades the performance of the solution. Data updates, either scheduled or triggered by environment change, might be needed. However, this is cumbersome and impractical over a long period.

 
 
 Device Heterogeneity: 
In the exact same physical location, two different devices might observe the same feature differently. This is usually due to the effective RF characteristics of the devices. Different devices might have different antenna design, RF components, or packaging, which all impact the received signal. It is difficult to predict what device the user might use, and in particular it is quite possible that the user uses a different device from the one used in collecting the dataset to train the model.

 
 
 Furthermore, different devices have different capabilities: while some devices can only read RSSI values, other might possibly be able to acquire CSI and other sensor readings. This is also rooted in the standards: different classes of devices are defined that may be equipped with different number of antennas or antennas architectures, and they might operate in different frequency bands.

 
 
 Body shadowing: 
Modern localization solutions are meant for mobile targets. Depending on the orientation of the user and the how the device is carried, the wireless signal could be occasionally shadowed by the user’s body or hands. In fact, it is reported that body shadowing could result in as much as 10 10 dB loss in the received power; these values depend on body size, shape, distance and orientation and operation frequency [ 395 , 396 , 397 , 398 ] . This could alter or block certain features, and thus impact the range and distribution of the observed features. To demonstrate this with simple practical example: the constants calculated to normalize the input features, as part of preprocessing (see Sec. II-B ), will be significantly impacted by the extreme variation of the features.

 
 
 User privacy: 
To collect large datasets, a number of works suggest the use of crowdsourcing [ 71 , 125 , 100 , 277 , 68 ] . In addition to the concern of the quality of the collected data and methods to label the them, the privacy of users is a major issue. Users generally avoid sharing their location and movement trends (if they are given the option). Furthermore, storing the location information in database for a relatively long time could also elevate that concern. With advent of data driven (ML) solutions, utilizing users data is both opportunity and threat to users privacy, which could lead to stringent regulations.

 
 
 

#### VII-A 2 Possible Solutions and Research Directions

 
 One solution to some of the aforementioned challenges is to collect the labeled data in brute-force manner, i.e., repeated collection of the feature-labels points, several times, by different administrated users, at different orientations, using different devices. This is clearly cost-prohibitive. Which motivated a number of interesting solutions that was covered in Sec. V (mainly, from subsection V-B and onward), However, there many more interesting avenues to utilize the power of ML.

 
 • 
 
 New data augmentation techniques can be used to expand the dataset. Besides the classical techniques, e.g., adding noise or masking part of the features, the following can be good options.

 
 – 
 
 Augmenting sounding measurements with synthetic data.

 

 – 
 
 Utilize generative models to capture the possible mapping between the features and possible variations or different dimensions (in case of different features sets, e.g., from a set of possible CSI values from limited observations). A few papers have considered that, e.g., [ 90 , 139 , 177 , 138 ] . However, more research is still needed.

 

 
 

 • 
 
 Employing TL could allow to use synthetic data or to train the solution in areas where access to a large number of points is relatively easy. As discussed in Sec. V-D , a number of works have considered TL, but both theoretical and experimental studies are needed to understand the best practices and limitation of TL for localization. For instance, motivated by different classes of channel models (e.g., micro, macro cells, urban, industrial etc.) or the dominant source of shadowing, TL may be easier between systems that are in similar environments.

 

 • 
 
 Separable models, composed of both generic and device specific models. This could handle device heterogeneity and reduce the transfer of the explicit raw user data.

 

 • 
 
 Utilizing measurement-validated channel characteristics, i.e., expert knowledge, in semi-supervised learning. Research is needed on how the few labeled data points can be used along with the unlabeled data, e.g., improved graphs based on modern channel models and environment structure. Furthermore, additional research can be done to improve where and when to update the annotated data points.

 

 • 
 
 Although a good number of works have investigated feature representation (e.g., ratio or difference of RSSIs), dimensionality reductions, and networks setups to combat channel variation and feature shifts, e.g., [ 86 , 143 , 104 , 156 , 167 ] , further research is still needed. In particular, the vast majority of the work in this area was dedicated to simple features, which may not be applicable to heterogeneous and complex features, which might be more robust to channel variations. As in illustrative example that was introduced in Fig. 2 , dimensionality reduction of channel multi-path multi-path components (that vary relatively slower than RSSI) could reveal underlying manifolds that may capture the structure in the environment. Furthermore, understanding the relation between the observed features and the proposed embedding and how the channel variations impact them could be of a value.

 

 • 
 
 Use of FL. As discussed in Sec. V-E2 , one goal of this technique is to preserve users’ privacy. However, devices suffer from different local constrains and impairments. There are recent studies that investigate the different techniques to combine the data, and thus reduce the impact of device-specific issues. Additionally, the use of FL could encourage the user to engage in data sharing, as no raw data is being transferred. Given the constraints on location information and data sharing, localization driven advances in FL are desirable.

 

 
 
 
 
 

### VII-B Utilization of DL 

 

#### VII-B 1 Discussion

 
 The number of published studies of DL-based localization is growing fast. Although DL offers a powerful solution that can utilize large datasets and is reported to easily generalize (at least in other fields), they are usually complex with a large number of trainable parameters (could be millions). This results in the following drawbacks:

 
 • 
 
 Computational difficulty to train. They may need GPUs to speed the training process, which may not always be available or expensive to install.

 

 • 
 
 Deep models have a large number of operations; this limits their use in power-limited devices.

 

 • 
 
 While DL solutions flourish with large datasets, acquiring such data is difficult.

 

 • 
 
 Overfitting. While this is generally a concern in ML solution, the large number of parameters in DL can exacerbate the problem.

 

 
 In the literature, the reported accuracy of different ML-based localization methods ranges from sub-centimeter to tens of meters. However, it is difficult to judge the solutions based on the reported accuracy only, as a number of factors play into the goodness of the solution and its practicality. In Fig. 10 and Fig. 9 we plot the reported localization error (in meter) vs dataset size and point resolution (in meter), respectively. For better comparability, we restricted the data to supervised learning in indoor environments. Usually, we are looking for solutions with small localization errors. However, the dataset plays a major role in the performance. Good solutions could achieve good accuracy with small examples. They could also generalize well even with large point separation. However, we do not see that for all the solutions. Rather, we notice that even with large datasets with DL solutions, large localization errors are observed in Fig. 9 . Similarly, in Fig. 10 we can observe that large errors are reported even at small separation distance. Additionally, in both figures, we notice that a number of DL solutions fall in the same region as the standard ML solutions. This should trigger the question of whether there is always a need for DL solutions. Nevertheless, we notice a number of DL solutions present in the right lower quadrant of Fig. 9 and the left lower quadrant of Fig. 10 , which confirms the potential advantages of DL when large datasets are available and better discriminative ability when points become close to one another.

 
 
 

#### VII-B 2 Possible Solutions and Research Directions

 
 Thus it is always recommended to test the performance of DL-solutions against standard simple ML solutions. Moreover, research should consider the offered improvement against the model complexity. An urgent research item is developing a wide range of datasets that cover different possible features and environments, which will allow fair comparative studies of the proposed solution. For the same reason, researchers are encouraged to publish their trained models online.

 
 
 Figure 9: Reported localization error in meter plotted vs the size of the used dataset. Solutions that use small datasets and achieve good results could indicate efficient performance. Small errors with larger datasets could suggest more discriminative power. Note these are not conclusive metrics as good performance in large datasets could be due to over sampling the area 
 
 
 Figure 10: Reported localization error in meters plotted vs the resolution of the points in the training dataset (in meters). Points below slope = 1 could suggest good regression and generalization abilities. The overall performance against density serve as an indicator of the cost of the solution (assuming similar set of features are used). Solutions that have small localization errors and use examples with small separation distance are preferred for practical applications. 
 
 
 
 

### VII-C System Reliability 

 

#### VII-C 1 Discussion

 
 In several proposed solutions, the device collects a large number of observations and sends them to the localization unit (e.g., BS or APs) for processing. Alternatively, the localization unit observes the features and processes them. With the expected increased demand for proximity and localization services in addition to the features complexity, the overhead of transmitting the observations and the processing shall increase, which may overwhelm the infrastructure. This has an impact on the localization and possibly the primary services, e.g., wireless communication.

 
 
 

#### VII-C 2 Possible Solutions and Research Directions

 
 
 • 
 
 Systematic studies of overhead and processing of the solutions are required. The impact of spectrum efficiency in wireless systems can be also investigated. Since location aided wireless communication is becoming prevalent, metrics that capture the cost and gain of use of location information are also interesting directions.

 

 • 
 
 Joint localization can alleviate the possible degradation of location estimates, as target channel observations may be correlated. With efficient ML-based implementation, joint localization can reduce processing time and improve the location estimate.

 

 • 
 
 Low dimensional feature extraction at the target end. To reduce the overhead, only compressed feature representations can be transmitted to the localization unit.

 

 • 
 
 ML-based localization at the target end, which can minimize both overhead and processing overload at the localization unit, can also help preserving users’ privacy. A relaxation of this is to implement part of the solution at the target end, which may be viewed as a generalization of the previous point. Few works have explicitly considered that, such as [ 240 ] , which proposes ”distributed CNN”, extracting latent variables locally before the compressed representation of CIR is passed to the central localization unit; different from FL, the training is done end-to-end before the part of the solution is distributed. Still much work is needed in this important direction.

 

 • 
 
 Cooperative localization, where the devices can share information that may improve location estimates. This can be used along the line of the points above, where the targets can locally use their neighbours’ estimates or features to improve their predictions. The large body of ML-based cooperative localization is based on range estimates, then applying MDS techniques (See Sec. V-B ); however, with the availability of rich features, solutions beyond range estimates can be proposed.

 

 
 
 
 
 

### VII-D Emerging applications and technologies: 

 
 The demand for RF-based localization has increased dramatically due to the recent advances in wireless communication systems and the rise of new applications. These have raised the bar of the expected localization accuracy. A few emerging applications and technologies are listed below.

 
 • 
 
 Accurate location information for vehicles; this is especially needed for smart and autonomous vehicles, where a few meters of localization errors may not be acceptable. What makes this application unique are (i) the environment is highly dynamic, (ii) communication is usually over low rate connections (e.g., IEEE 802.11p), (iii) availability of a wide range of side information types such as images and other sensors reading (iv) computational power may not be a constraint.

 

 • 
 
 Mission-critical applications in highly dynamic and harsh environments. This can be viewed as a generalization of the above point. One example may be a factory, where wireless channels suffer from many (possibly dynamic) scatterers. Furthermore, excessive location errors can have catastrophic consequences. A few studies have considered ML based localization in industrial environment, e.g., [ 83 , 288 , 224 , 223 , 240 , 399 ] . However, ML-based methods to combat interference and provide estimates guarantees e.g., through adversarial networks could be studied [ 24 ] .

 

 • 
 
 Localization in IoT, were the services cover applications with heterogeneous devices and wireless standards, This calls for flexible ML-based solutions that take the various constraints at the wide range of devices into account. Constraints include limitations on observable features, processing power, and acceptable localization error. An important goal is designing good embedding that captures the relation between the different devices, e.g., graph based solution through the graph neural networks [ 400 ] .

 

 • 
 
 Massive MIMO. Although a number of approaches have already been proposed, there are a number of challenges that should be addressed. They include the excessive overhead for feature feedback (see also Sec. VII-C ), or the needed beam-search along with possible beam outdatedness and misalignments. Massive MIMO is expected to be deployed in mmWave communication systems, which have multi-GHz bandwidth, method for efficient utilization of the large bandwidth or number of sub-carriers are needed.

 

 • 
 
 Utilizing RF-signals along with side information for localization related problems, such as Simultaneous Localization and Mapping (SLAM) and Radar. Both these areas have an overlap with localization problems. In SLAM the goal is to track a moving target (agent as referred to in robotics) along with the construction of the map of the environment. Typically, different types of data are used to solve this problem, such as Inertial Sensor readings and images, i.e., the side information listed in Sec. IV-D . Solutions that implicitly construct the map (as in Sec. V-B ), and DL solutions that efficiently extract the RF-features especially in high bandwidth systems (e.g., UWB and high frequency ranges,i.e., mmWave and beyond), are good research directions in this field.
In the Radar case, the goal is to detect, locate and identify the targets, which usually utilize different types of data as well. RF-based passive localization that we covered in this survey is a simpler version of that. Researchers can exploit some of the techniques used in the radar field to enhance the localization.

 
 
 ML techniques have been used in both fields, for surveys see [ 22 ] and [ 4 ] . With anticipated (DL driven) advances of RF-based localization, enhancements to both fields can be introduced.

 

 
 
 
 
 

## VIII Conclusions 

 
 In this paper, we surveyed ML-based localization solutions. We focused on systems that use different attributes of the RF signals to localize their targets. The survey spans different aspects of ML-based localization: the system architectures, the used RF features, the ML methods, and the data acquisitions. Throughout the survey, we maintained structured reference lists for the relevant aspects of the solutions. To make the presentation accessible to readers from different backgrounds, we presented a concise review of the main aspects of ML and wireless channels.

 
 
 Based on the surveyed solutions, we identified different challenges and research directions when applying ML in localization. In particular, as data-driven solutions, enabling ML-based localization requires efficient methods to collect and maintain the datasets, which are usually subject to the environment’s dynamics and the acquisition systems. One of the suggested solutions is based on augmentation techniques and deep generative models that may enrich the datasets, where the latter may learn the true distribution of the data and its evolution. Alternatively, methods that capture the fundamental properties of wireless channels and utilize TL and Semi-supervised learning can be used. The overhead of feature transfer and the complexity of DL are of concern, methods to distribute the learning (e.g., using FL or separable models) are of interest. Furthermore, datasets with rich features are needed to enable researchers to study and compare their solutions. Overall, ML-based localization has many interesting research avenues as new wireless systems, standards, and applications are being progressively introduced.

 Acknowledgements: The authors thank Prof. Mahdi Soltonkotabi for helpful discussions and critical reading of the manuscript.

 
 
 Table VIII: Used Acronyms 
 
 
 
 
 
 Acronym 
 | 
 
 
 Explanation 
 | 

 
 
 
 
 
 AE 
 | 
 
 
 Auto encoder 
 | 

 
 
 
 AP 
 | 
 
 
 Access Point 
 | 

 
 
 
 BLE 
 | 
 
 
 Bluetooth Low Energy 
 | 

 
 
 
 BS 
 | 
 
 
 Base-station 
 | 

 
 
 
 CFR 
 | 
 
 
 channel frequency response 
 | 

 
 
 
 CIR 
 | 
 
 
 channel impulse response 
 | 

 
 
 
 CNN 
 | 
 
 
 Convolution NN 
 | 

 
 
 
 CSI 
 | 
 
 
 Channel State Information 
 | 

 
 
 
 DBN 
 | 
 
 
 Deep Belief Networks 
 | 

 
 
 
 DL 
 | 
 
 
 Deep Learning 
 | 

 
 
 
 ELM 
 | 
 
 
 Extreme Learning Machine 
 | 

 
 
 
 EM 
 | 
 
 
 Expectation maximization 
 | 

 
 
 
 FL 
 | 
 
 
 Federated Learning 
 | 

 
 
 
 GMM 
 | 
 
 
 Gaussian mixture model 
 | 

 
 
 
 GP 
 | 
 
 
 Gaussian Process 
 | 

 
 
 
 GNNS 
 | 
 
 
 Global Navigation Satellite Systems 
 | 

 
 
 
 GRU 
 | 
 
 
 Gated Recurrent Units 
 | 

 
 
 
 HMM 
 | 
 
 
 Hidden Markov Model 
 | 

 
 
 
 IoT 
 | 
 
 
 Internet of Things 
 | 

 
 
 
 KL divergence 
 | 
 
 
 Kullback-Leibler divergence 
 | 

 
 
 
 KNN 
 | 
 
 
 K-Nearest Neighbors 
 | 

 
 
 
 LDA 
 | 
 
 
 Linear Discriminant Analysis 
 | 

 
 
 
 LOS 
 | 
 
 
 Line of Sight 
 | 

 
 
 
 LoRaWAN 
 | 
 
 
 Long Range WAN (a protocol) 
 | 

 
 
 
 LSTM 
 | 
 
 
 Long-Short-Term Memory 
 | 

 
 
 
 MDN 
 | 
 
 
 multi density network 
 | 

 
 
 
 ML 
 | 
 
 
 Machine Learning 
 | 

 
 
 
 mmWave 
 | 
 
 
 millimeter-wave 
 | 

 
 
 
 MS 
 | 
 
 
 Mobile station 
 | 

 
 
 
 NN 
 | 
 
 
 Neural Network 
 | 

 
 
 
 PCA 
 | 
 
 
 Principal Component Analysis 
 | 

 
 
 
 PDP 
 | 
 
 
 Power Delay Profile 
 | 

 
 
 
 RBF 
 | 
 
 
 Radial Basis Function 
 | 

 
 
 
 RF 
 | 
 
 
 Radio Frequency 
 | 

 
 
 
 RMSE 
 | 
 
 
 Root Mean Square Error 
 | 

 
 
 
 RNN 
 | 
 
 
 Recurrent NN 
 | 

 
 
 
 RP 
 | 
 
 
 Reference Point 
 | 

 
 
 
 RSRP 
 | 
 
 
 Reference Signal Received Power 
 | 

 
 
 
 RSRQ 
 | 
 
 
 Reference Signal Received Quality 
 | 

 
 
 
 RSS 
 | 
 
 
 Received Signal Strength 
 | 

 
 
 
 RSU 
 | 
 
 
 Road Side Units 
 | 

 
 
 
 SINR 
 | 
 
 
 Signal to interference and noise power ratio 
 | 

 
 
 
 SVM 
 | 
 
 
 Support Vector Machine 
 | 

 
 
 
 TDOA 
 | 
 
 
 Time difference of arrival 
 | 

 
 
 
 TL 
 | 
 
 
 Transfer Learning 
 | 

 
 
 
 ToA 
 | 
 
 
 Time of arrival 
 | 

 
 
 
 UAV 
 | 
 
 
 Unmanned Aerial Vehicle 
 | 

 
 
 
 UE 
 | 
 
 
 User Equipment 
 | 

 
 
 
 WAN 
 | 
 
 
 Wireless Area Network 
 | 

 
 
 
 WSN 
 | 
 
 
 Wireless sensor network 
 | 

 

 
 
 

## References

 
 
 [1] 
 
R. Zekavat and R. M. Buehrer, Handbook of position location: Theory,
practice and advances , 2nd ed. John
Wiley Sons, 2013.

 

 
 [2] 
 
S. Gezici, Z. Tian, G. B. Giannakis, H. Kobayashi, A. F. Molisch, H. V. Poor,
and Z. Sahinoglu, “Localization via ultra-wideband radios: a look at
positioning aspects for future sensor networks,” IEEE signal
processing magazine , vol. 22, no. 4, pp. 70–84, 2005.

 

 
 [3] 
 
W. I. Remcom,
“https://www.gartner.com/en/newsroom/press-releases/2018-03-20-gartner-highlights-10-uses-for-ai-powered-smartphones,”
online, accessed: December 2020.

 

 
 [4] 
 
P. Lang, X. Fu, M. Martorella, J. Dong, R. Qin, X. Meng, and M. Xie, “A
comprehensive survey of machine learning applied to radar signal
processing,” arXiv preprint arXiv:2009.13702 , 2020.

 

 
 [5] 
 
F. Zafari, A. Gkelias, and K. K. Leung, “A survey of indoor localization
systems and technologies,” IEEE Communications Surveys Tutorials ,
vol. 21, no. 3, pp. 2568–2599, 2019.

 

 
 [6] 
 
H. Liu, H. Darabi, P. Banerjee, and J. Liu, “Survey of wireless indoor
positioning techniques and systems,” IEEE Transactions on Systems,
Man, and Cybernetics, Part C (Applications and Reviews) , vol. 37, no. 6, pp.
1067–1080, 2007.

 

 
 [7] 
 
G. Deak, K. Curran, and J. Condell, “A survey of active and passive indoor
localisation systems,” Computer Communications , vol. 35, no. 16, pp.
1939–1954, 2012.

 

 
 [8] 
 
Y. Gu, A. Lo, and I. Niemegeers, “A survey of indoor positioning systems for
wireless personal networks,” IEEE Communications surveys 
tutorials , vol. 11, no. 1, pp. 13–32, 2009.

 

 
 [9] 
 
F. Wen, H. Wymeersch, B. Peng, W. P. Tay, H. C. So, and D. Yang, “A survey on
5G massive MIMO localization,” Digital Signal Processing ,
vol. 94, pp. 21–28, 2019.

 

 
 [10] 
 
J. A. del Peral-Rosado, R. Raulefs, J. A. López-Salcedo, and
G. Seco-Granados, “Survey of cellular mobile radio localization methods:
From 1G to 5G,” IEEE Communications Surveys Tutorials ,
vol. 20, no. 2, pp. 1124–1148, 2017.

 

 
 [11] 
 
S. He and S.-H. G. Chan, “Wi-Fi fingerprint-based indoor positioning: Recent
advances and comparisons,” IEEE Communications Surveys Tutorials ,
vol. 18, no. 1, pp. 466–490, 2015.

 

 
 [12] 
 
X. Lin, J. Bergman, F. Gunnarsson, O. Liberg, S. M. Razavi, H. S. Razaghi,
H. Rydn, and Y. Sui, “Positioning for the internet of things: A 3GPP
perspective,” IEEE Communications Magazine , vol. 55, no. 12, pp.
179–185, 2017.

 

 
 [13] 
 
P. Davidson and R. Piché, “A survey of selected indoor positioning methods
for smartphones,” IEEE Communications Surveys Tutorials , vol. 19,
no. 2, pp. 1347–1370, 2016.

 

 
 [14] 
 
A. F. G. Ferreira, D. M. A. Fernandes, A. P. Catarino, and J. L. Monteiro,
“Localization and positioning systems for emergency responders: A survey,”
 IEEE Communications Surveys Tutorials , vol. 19, no. 4, pp.
2836–2870, 2017.

 

 
 [15] 
 
A. Tahat, G. Kaddoum, S. Yousefi, S. Valaee, and F. Gagnon, “A look at the
recent wireless positioning techniques with a focus on algorithms for moving
receivers,” IEEE Access , vol. 4, pp. 6652–6680, 2016.

 

 
 [16] 
 
R. C. Shit, S. Sharma, D. Puthal, P. James, B. Pradhan, A. van Moorsel, A. Y.
Zomaya, and R. Ranjan, “Ubiquitous localization (UbiLoc): a survey and
taxonomy on device free localization for smart world,” IEEE
Communications Surveys Tutorials , vol. 21, no. 4, pp. 3532–3564, 2019.

 

 
 [17] 
 
Z. Yang, Z. Zhou, and Y. Liu, “From RSSI to CSI: Indoor localization via
channel response,” ACM Computing Surveys (CSUR) , vol. 46, no. 2, pp.
1–32, 2013.

 

 
 [18] 
 
W. Liu, Q. Cheng, Z. Deng, H. Chen, X. Fu, X. Zheng, S. Zheng, C. Chen, and
S. Wang, “Survey on CSI-based indoor positioning systems and recent
advances,” in 2019 International Conference on Indoor Positioning and
Indoor Navigation (IPIN) . IEEE, pp.
1–8.

 

 
 [19] 
 
Z. Yang, C. Wu, Z. Zhou, X. Zhang, X. Wang, and Y. Liu, “Mobility increases
localizability: A survey on wireless indoor localization using inertial
sensors,” ACM Computing Surveys (Csur) , vol. 47, no. 3, pp. 1–34,
2015.

 

 
 [20] 
 
R. Harle, “A survey of indoor inertial positioning systems for pedestrians,”
 IEEE Communications Surveys Tutorials , vol. 15, no. 3, pp.
1281–1293, 2013.

 

 
 [21] 
 
N. Saeed, H. Nam, T. Y. Al-Naffouri, and M.-S. Alouini, “A state-of-the-art
survey on multidimensional scaling-based localization techniques,”
 IEEE Communications Surveys Tutorials , vol. 21, no. 4, pp.
3565–3583, 2019.

 

 
 [22] 
 
C. Chen, B. Wang, C. X. Lu, N. Trigoni, and A. Markham, “A survey on deep
learning for localization and mapping: Towards the age of spatial machine
intelligence,” arXiv preprint arXiv:2006.12567 , 2020.

 

 
 [23] 
 
K. P. Murphy, Machine learning: a probabilistic perspective , 2012.

 

 
 [24] 
 
I. Goodfellow, Y. Bengio, and A. Courville, Deep learning , 2016, vol. 1.

 

 
 [25] 
 
G. James, D. Witten, T. Hastie, and R. Tibshirani, An introduction to
statistical learning . Springer, 2013,
vol. 112.

 

 
 [26] 
 
Y. S. Abu-Mostafa, M. Magdon-Ismail, and H.-T. Lin, Learning from
data . AMLBook New York, NY, USA:,
2012, vol. 4.

 

 
 [27] 
 
D. Koller and N. Friedman, Probabilistic graphical models: principles and
techniques . MIT press, 2009.

 

 
 [28] 
 
A. Géron, Hands-on machine learning with Scikit-Learn, Keras, and
TensorFlow: Concepts, tools, and techniques to build intelligent
systems . O’Reilly Media, 2019.

 

 
 [29] 
 
S. T. Roweis and L. K. Saul, “Nonlinear dimensionality reduction by locally
linear embedding,” science , vol. 290, no. 5500, pp. 2323–2326, 2000.

 

 
 [30] 
 
L. v. d. Maaten and G. Hinton, “Visualizing data using t-SNE,”
 Journal of machine learning research , vol. 9, no. Nov, pp. 2579–2605,
2008.

 

 
 [31] 
 
D. Burghal, R. Wang, and A. F. Molisch, “Band assignment in dual band systems:
A learning-based approach,” in MILCOM 2018-2018 IEEE Military
Communications Conference (MILCOM) . IEEE, 2018, pp. 7–13.

 

 
 [32] 
 
K. Hornik, “Approximation capabilities of multilayer feedforward networks,”
 Neural networks , vol. 4, no. 2, pp. 251–257, 1991.

 

 
 [33] 
 
A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with
deep convolutional neural networks,” Communications of the ACM ,
vol. 60, no. 6, pp. 84–90, 2017.

 

 
 [34] 
 
I. Sutskever, O. Vinyals, and Q. V. Le, “Sequence to sequence learning with
neural networks,” in Advances in neural information processing
systems , 2014, pp. 3104–3112.

 

 
 [35] 
 
X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W.-K. Wong, and W.-c. Woo,
“Convolutional lstm network: A machine learning approach for precipitation
nowcasting,” Advances in neural information processing systems ,
vol. 28, pp. 802–810, 2015.

 

 
 [36] 
 
A. F. Molisch, Wireless Communications . IEEE Press - Wiley; 2 edition, 2011.

 

 
 [37] 
 
M. Steinbauer, A. F. Molisch, and E. Bonek, “The double-directional radio
channel,” IEEE Antennas and propagation Magazine , vol. 43, no. 4, pp.
51–63, 2001.

 

 
 [38] 
 
A. F. Molisch, “Ultra-wide-band propagation channels,” Proceedings of
the IEEE , vol. 97, no. 2, pp. 353–371, 2009.

 

 
 [39] 
 
A. Richter, “Estimation of radio channel parameters: Models and
algorithms.” ISLE, 2005.

 

 
 [40] 
 
M. Shafi, M. Zhang, A. L. Moustakas, P. J. Smith, A. F. Molisch, F. Tufvesson,
and S. H. Simon, “Polarized MIMO channels in 3-D: models, measurements
and mutual information,” IEEE Journal on Selected Areas in
Communications , vol. 24, no. 3, pp. 514–527, 2006.

 

 
 [41] 
 
A. M. Sayeed, “Deconstructing multiantenna fading channels,” IEEE
Transactions on Signal processing , vol. 50, no. 10, pp. 2563–2579, 2002.

 

 
 [42] 
 
K. Witrisal, P. Meissner, E. Leitinger, Y. Shen, C. Gustafson, F. Tufvesson,
K. Haneda, D. Dardari, A. F. Molisch, A. Conti et al. , “High-accuracy
localization for assisted living: 5G systems will turn multipath channels
from foe to friend,” IEEE Signal Processing Magazine , vol. 33, no. 2,
pp. 59–70, 2016.

 

 
 [43] 
 
G. Calcev, D. Chizhik, B. Goransson, S. Howard, H. Huang, A. Kogiantis, A. F.
Molisch, A. L. Moustakas, D. Reed, and H. Xu, “A wideband spatial channel
model for system-wide simulations,” IEEE Transactions on Vehicular
Technology , vol. 56, no. 2, pp. 389–403, 2007.

 

 
 [44] 
 
3GPP, “Study on channel model for frequencies from 0.5 to 100 GHz,”
 TR 38.901 version 14.0.0 , 2017.

 

 
 [45] 
 
S. Jaeckel, L. Raschkowski, K. Börner, and L. Thiele, “Quadriga: A 3-D
multi-cell channel model with time evolution for enabling virtual field
trials,” IEEE Transactions on Antennas and Propagation , vol. 62,
no. 6, pp. 3242–3256, 2014.

 

 
 [46] 
 
J. Karedal, S. Wyne, P. Almers, F. Tufvesson, and A. F. Molisch, “UWB
channel measurements in an industrial environment,” in IEEE Global
Telecommunications Conference, 2004. GLOBECOM’04. , vol. 6. IEEE, 2004, pp. 3511–3516.

 

 
 [47] 
 
V. Kristem, S. Niranjayan, S. Sangodoyin, and A. F. Molisch, “Experimental
determination of UWB ranging errors in an outdoor environment,” in
 2014 ieee international conference on communications (icc) . IEEE, 2014, pp. 4838–4843.

 

 
 [48] 
 
A. F. Molisch, K. Balakrishnan, C.-C. Chong, S. Emami, A. Fort, J. Karedal,
J. Kunisch, H. Schantz, U. Schuster, and K. Siwiak, “IEEE 802.15. 4a
channel model-final report,” IEEE P802 , vol. 15, no. 04, p. 0662,
2004.

 

 
 [49] 
 
T. Van Nguyen, Y. Jeong, H. Shin, and M. Z. Win, “Machine learning for
wideband localization,” IEEE Journal on Selected Areas in
Communications , vol. 33, no. 7, pp. 1357–1380, 2015.

 

 
 [50] 
 
C. Huang, A. F. Molisch, R. He, R. Wang, P. Tang, B. Ai, and Z. Zhong,
“Machine learning-enabled LOS/NLOS identification for MIMO system in
dynamic environment,” IEEE Transactions on Wireless Communications ,
2020.

 

 
 [51] 
 
B. Peng, G. Seco-Granados, E. Steinmetz, M. Fröhle, and H. Wymeersch,
“Decentralized scheduling for cooperative localization with deep
reinforcement learning,” IEEE Transactions on Vehicular Technology ,
vol. 68, no. 5, pp. 4295–4305, 2019.

 

 
 [52] 
 
J. Curro, J. Raquet, and B. Borghetti, “Navigation using VLF signals with
artificial neural networks,” Navigation , vol. 65, 12 2018.

 

 
 [53] 
 
B. A. Akram, A. H. Akbar, and O. Shafiq, “Hybloc: Hybrid indoor Wi-Fi
localization using soft clustering-based random decision forest ensembles,”
 IEEE Access , vol. 6, pp. 38 251–38 272, 2018.

 

 
 [54] 
 
K. M. Chen, R. Y. Chang, and S. Liu, “Interpreting convolutional neural
networks for device-free Wi-Fi fingerprinting indoor localization via
information visualization,” IEEE Access , vol. 7, pp.
172 156–172 166, 2019.

 

 
 [55] 
 
X. Dang, X. Tang, Z. Hao, and Y. Liu, “A device-free indoor localization
method using CSI with Wi-Fi signals,” Sensors , vol. 19, no. 14,
p. 3233, 2019.

 

 
 [56] 
 
L. Zhao, H. Huang, S. Ding, and X. Li, “An accurate and efficient
device-free localization approach based on Gaussian Bernoulli restricted
Boltzmann machine,” in 2018 IEEE International Conference on
Systems, Man, and Cybernetics (SMC) , 2018, pp. 2323–2328.

 

 
 [57] 
 
Y. Zhang, D. Li, and Y. Wang, “An indoor passive positioning method
using CSI fingerprint based on Adaboost,” IEEE Sensors Journal ,
vol. 19, no. 14, pp. 5792–5800, 2019.

 

 
 [58] 
 
S. Fang, C. Li, W. Lu, Z. Xu, and Y. Chien, “Enhanced device-free
human detection: Efficient learning from phase and amplitude of channel state
information,” IEEE Transactions on Vehicular Technology , vol. 68,
no. 3, pp. 3048–3051, 2019.

 

 
 [59] 
 
L. Zhao, C. Su, D. Zeyang, H. Huang, S. Ding, X. Huang, and Z. Han, “Indoor
device-free passive localization with DCNN for location-based services,”
 The Journal of Supercomputing , 12 2019.

 

 
 [60] 
 
Z. Wu, L. Jiang, Z. Jiang, B. Chen, K. Liu, Q. Xuan, and Y. Xiang, “Accurate
indoor localization based on CSI and visibility graph,” Sensors ,
vol. 18, no. 8, p. 2549, 2018.

 

 
 [61] 
 
T. F. Sanam and H. Godrich, “An improved CSI based device free indoor
localization using machine learning based classification approach,” in
 2018 26th European Signal Processing Conference (EUSIPCO) . IEEE, 2018, pp. 2390–2394.

 

 
 [62] 
 
L. Zhao, H. Huang, X. Li, S. Ding, H. Zhao, and Z. Han, “An accurate and
robust approach of device-free localization with convolutional autoencoder,”
 IEEE Internet of Things Journal , vol. 6, no. 3, pp. 5825–5840, 2019.

 

 
 [63] 
 
Y. Guo, D. Yu, and N. Li, “Exploiting fine-grained subcarrier information for
device-free localization in wireless sensor networks,” Sensors ,
vol. 18, p. 3110, 09 2018.

 

 
 [64] 
 
Z. Yang, C. Liu, and L. Jin, “A clustering-based algorithm for device-free
localization in IoT,” in 2018 IEEE 4th International Conference on
Computer and Communications (ICCC) . IEEE, 2018, pp. 769–773.

 

 
 [65] 
 
Z. Chen and J. Wang, “ES-DPR: A DOA-based method for passive localization in
indoor environments,” Sensors , vol. 19, no. 11, p. 2482, 2019.

 

 
 [66] 
 
J. Yoo and J. Park, “Indoor localization based on Wi-Fi received signal
strength indicators: Feature extraction, mobile fingerprinting, and
trajectory learning,” Applied Sciences , vol. 9, p. 3930, 09 2019.

 

 
 [67] 
 
I. Saffar, M. L. A. Morel, K. D. Singh, and C. Viho, “Machine learning
with partially labeled data for indoor outdoor detection,” in 2019
16th IEEE Annual Consumer Communications Networking Conference (CCNC) , 2019,
pp. 1–8.

 

 
 [68] 
 
C. Wu, Z. Yang, and Y. Liu, “Smartphones based crowdsourcing for indoor
localization,” IEEE Transactions on Mobile Computing , vol. 14, no. 2,
pp. 444–457, 2015.

 

 
 [69] 
 
K. Chow, S. He, J. Tan, and S. . G. Chan, “Efficient locality
classification for indoor fingerprint-based systems,” IEEE
Transactions on Mobile Computing , vol. 18, no. 2, pp. 290–304, 2019.

 

 
 [70] 
 
H. Aly and A. Agrawala, “Hapi,” Proceedings of the 15th EAI
International Conference on Mobile and Ubiquitous Systems: Computing,
Networking and Services , Nov 2018. [Online]. Available:
 http://dx.doi.org/10.1145/3286978.3286980 

 

 
 [71] 
 
J. Yoo, K. H. Johansson, and H. Jin Kim, “Indoor localization without a
prior map by trajectory learning from crowdsourced measurements,” IEEE
Transactions on Instrumentation and Measurement , vol. 66, no. 11, pp.
2825–2835, 2017.

 

 
 [72] 
 
Z. Yang, C. Wu, and Y. Liu, “Locating in fingerprint space: wireless indoor
localization with little human intervention,” in Proceedings of the
18th annual international conference on Mobile computing and networking ,
2012, pp. 269–280.

 

 
 [73] 
 
K. S. Kim, S. Lee, and K. Huang, “A scalable deep neural network architecture
for multi-building and multi-floor indoor localization based on Wi-Fi
fingerprinting,” Big Data Analytics , vol. 3, no. 1, p. 4, 2018.

 

 
 [74] 
 
K. S. Kim, “Hybrid building/floor classification and location coordinates
regression using a single-input and multi-output deep neural network for
large-scale indoor localization based on Wi-Fi fingerprinting,” in
 2018 Sixth International Symposium on Computing and Networking
Workshops (CANDARW) . IEEE, 2018, pp.
196–201.

 

 
 [75] 
 
A. Ç. Seçkin and A. Coşkun, “Hierarchical fusion of machine
learning algorithms in indoor positioning and localization,” Applied
Sciences , vol. 9, no. 18, p. 3665, 2019.

 

 
 [76] 
 
Y. Oussar, I. Ahriz, B. Denby, and G. Dreyfus, “Indoor localization based on
cellular telephony RSSI fingerprints containing very large numbers of
carriers,” EURASIP Journal on Wireless Communications and Networking ,
vol. 2011, no. 1, p. 81, 2011.

 

 
 [77] 
 
E. Schmidt, D. Inupakutika, R. Mundlamuri, and D. Akopian, “SDR-Fi:
Deep-learning-based indoor positioning via software-defined radio,”
 IEEE Access , vol. 7, pp. 145 784–145 797, 2019.

 

 
 [78] 
 
X. Nguyen, M. I. Jordan, and B. Sinopoli, “A kernel-based learning approach to
Ad Hoc sensor network localization,” ACM Transactions on Sensor
Networks (TOSN) , vol. 1, no. 1, pp. 134–152, 2005.

 

 
 [79] 
 
X. Guo, N. Ansari, L. Li, and H. Li, “Indoor localization by fusing a
group of fingerprints based on random forests,” IEEE Internet of
Things Journal , vol. 5, no. 6, pp. 4686–4698, 2018.

 

 
 [80] 
 
X. Yang, F. Zhao, and T. Chen, “NLOS identification for UWB localization
based on import vector machine,” AEU - International Journal of
Electronics and Communications , vol. 87, 02 2018.

 

 
 [81] 
 
M. Nuno, H. Herrera-Rivas, C. Torres-Huitzil, H. Marin, and Y. Coronado-Pérez,
“On-device learning of indoor location for WiFi fingerprint approach,”
 Sensors , vol. 18, p. 2202, 07 2018.

 

 
 [82] 
 
Y. Chen, C. Hsu, C. Huang, and H. Hung, “Outdoor localization for
LoRaWans using semi-supervised transfer learning with grid segmentation,”
in 2019 IEEE VTS Asia Pacific Wireless Communications Symposium
(APWCS) , 2019, pp. 1–5.

 

 
 [83] 
 
B. Silva, R. dos Santos, and G. P. Hancke, “Towards non-line-of-sight
ranging error mitigation in industrial wireless sensor networks,” in
 IECON 2016 - 42nd Annual Conference of the IEEE Industrial Electronics
Society , 2016, pp. 5687–5692.

 

 
 [84] 
 
Y. Y. Munaye, H.-P. Lin, A. B. Adege, and G. B. Tarekegn, “Uav positioning
for throughput maximization using deep learning approaches,” Sensors ,
vol. 19, no. 12, p. 2775, 2019.

 

 
 [85] 
 
D. Sikeridis, B. P. Rimal, I. Papapanagiotou, and M. Devetsikiotis,
“Unsupervised crowd-assisted learning enabling location-aware facilities,”
 IEEE Internet of Things Journal , vol. 5, no. 6, pp. 4699–4713, 2018.

 

 
 [86] 
 
F. Zhao, T. Huang, and D. Wang, “A probabilistic approach for WiFi
fingerprint localization in severely dynamic indoor environments,”
 IEEE Access , vol. 7, pp. 116 348–116 357, 2019.

 

 
 [87] 
 
X. Wang and Y. Feng, “An ensemble learning algorithm for indoor
localization,” in 2018 IEEE 4th International Conference on Computer
and Communications (ICCC) , 2018, pp. 774–778.

 

 
 [88] 
 
P. Dai, Y. Yang, M. Wang, and R. Yan, “Combination of DNN and improved KNN
for indoor location fingerprinting,” Wireless Communications and
Mobile Computing , vol. 2019, pp. 1–9, 03 2019.

 

 
 [89] 
 
J. L. V. Carrera, Z. Zhao, T. Braun, H. Luo, and F. Zhao, “Discriminative
learning-based smartphone indoor localization,” 2018.

 

 
 [90] 
 
H. Rizk, A. Shokry, and M. Youssef, “Effectiveness of data augmentation in
cellular-based localization using deep learning,” 2019 IEEE Wireless
Communications and Networking Conference (WCNC) , Apr 2019. [Online].
Available: http://dx.doi.org/10.1109/WCNC.2019.8886005 

 

 
 [91] 
 
S. Timotheatos, G. Tsagkatakis, P. Tsakalides, and P. E. Trahanias, “Feature
extraction and learning for RSSI based indoor device localization,” in
 ESANN , 2017.

 

 
 [92] 
 
G. Apostolo, I. Sampaio, and J. Viterbo, “Feature selection on database
optimization for Wi-Fi fingerprint indoor positioning,” Procedia
Computer Science , vol. 159, pp. 251–260, 01 2019.

 

 
 [93] 
 
L. Zhang and H. Wang, “Fingerprinting-based indoor localization with
relation learning network,” in 2019 IEEE/CIC International Conference
on Communications in China (ICCC) , 2019, pp. 541–544.

 

 
 [94] 
 
I. Ahriz, Y. Oussar, B. Denby, and G. Dreyfus, “Full-band GSM fingerprints
for indoor localization using a machine learning approach,”
 International Journal of Navigation and Observation , vol. 2010, 05
2010.

 

 
 [95] 
 
L. Li, W. Yang, and G. Wang, “HIWL: An unsupervised learning algorithm
for indoor wireless localization,” in 2013 12th IEEE International
Conference on Trust, Security and Privacy in Computing and Communications ,
2013, pp. 1747–1753.

 

 
 [96] 
 
Y. Feng, J. Minghua, L. Jing, Q. Xiao, H. Ming, P. Tao, and
H. Xinrong, “Improved AdaBoost-based fingerprint algorithm for wifi
indoor localization,” in 2014 IEEE 7th Joint International Information
Technology and Artificial Intelligence Conference , 2014, pp. 16–19.

 

 
 [97] 
 
P. Roy, C. Chowdhury, D. Ghosh, and S. Bandyopadhyay, “JUIndoorLoc: A
ubiquitous framework for smartphone-based indoor localization subject to
context and device heterogeneity,” Wireless Personal Communications ,
vol. 106, 02 2019.

 

 
 [98] 
 
X. Guo, L. Wang, L. Li, and N. Ansari, “Transferred knowledge aided
positioning via global and local structural consistency constraints,”
 IEEE Access , vol. 7, pp. 32 102–32 117, 2019.

 

 
 [99] 
 
X. Wang, “WiFi fingerprinting based indoor localization: When CSI tensor
meets deep residual sharing learning,” 2017.

 

 
 [100] 
 
S. Chunjing and J. Wang, “WLAN fingerprint indoor positioning strategy based
on implicit crowdsourcing and semi-supervised learning,” ISPRS
International Journal of Geo-Information , vol. 6, p. 356, 11 2017.

 

 
 [101] 
 
N. Anzum, S. F. Afroze, and A. Rahman, “Zone-based indoor localization
using neural networks: A view from a real testbed,” in 2018 IEEE
International Conference on Communications (ICC) , 2018, pp. 1–7.

 

 
 [102] 
 
P. Cottone, S. Gaglio, G. L. Re, and M. Ortolani, “A machine learning approach
for user localization exploiting connectivity data,” Engineering
Applications of Artificial Intelligence , vol. 50, pp. 125–134, 2016.

 

 
 [103] 
 
Z. E. Khatab, A. Hajihoseini, and S. A. Ghorashi, “A fingerprint method for
indoor localization using autoencoder based deep extreme learning machine,”
 IEEE sensors letters , vol. 2, no. 1, pp. 1–4, 2017.

 

 
 [104] 
 
Y. Rezgui, L. Pei, X. Chen, F. Wen, and C. Han, “An efficient normalized rank
based SVM for room level indoor WiFi localization with diverse devices,”
 Mobile Information Systems , vol. 2017, 2017.

 

 
 [105] 
 
Z. Liu, B. Dai, X. Wan, and X. Li, “Hybrid wireless fingerprint indoor
localization method based on a convolutional neural network,”
 Sensors , vol. 19, no. 20, p. 4597, 2019.

 

 
 [106] 
 
H. Ahmadi and R. Bouallegue, “Exploiting machine learning strategies and
RSSI for localization in wireless sensor networks: A survey,” in
 2017 13th International Wireless Communications and Mobile Computing
Conference (IWCMC) . IEEE, 2017, pp.
1150–1154.

 

 
 [107] 
 
Y. Gu, Y. Chen, J. Liu, and X. Jiang, “Online deep intelligence for Wi-Fi
indoor localization,” in Adjunct Proceedings of the 2015 ACM
International Joint Conference on Pervasive and Ubiquitous Computing and
Proceedings of the 2015 ACM International Symposium on Wearable Computers ,
2015, pp. 29–32.

 

 
 [108] 
 
J. Yan, L. Zhao, J. Tang, Y. Chen, R. Chen, and L. Chen, “Hybrid kernel based
machine learning using received signal strength measurements for indoor
localization,” IEEE Transactions on Vehicular Technology , vol. 67,
no. 3, pp. 2824–2829, 2017.

 

 
 [109] 
 
A. H. Salamah, M. Tamazin, M. A. Sharkas, and M. Khedr, “An enhanced WiFi
indoor localization system based on machine learning,” in 2016
International Conference on Indoor Positioning and Indoor Navigation
(IPIN) . IEEE, 2016, pp. 1–8.

 

 
 [110] 
 
S. Zhang, J. Guo, N. Luo, L. Wang, W. Wang, and K. Wen, “Improving Wi-Fi
fingerprint positioning with a pose recognition-assisted SVM algorithm,”
 Remote Sensing , vol. 11, no. 6, p. 652, 2019.

 

 
 [111] 
 
A. Belmonte-Hernández, G. Hernández-Peñaloza, D. M. Gutiérrez,
and F. Álvarez, “SWiBluX: Multi-sensor deep learning fingerprint for
precise real-time indoor tracking,” IEEE Sensors Journal , vol. 19,
no. 9, pp. 3473–3486, 2019.

 

 
 [112] 
 
L. Li, X. Guo, and N. Ansari, “SmartLoc: Smart wireless indoor localization
empowered by machine learning,” IEEE Transactions on Industrial
Electronics , 2019.

 

 
 [113] 
 
K. N. R. S. V. Prasad, E. Hossain, and V. K. Bhargava, “Machine learning
methods for user positioning with uplink RSS in distributed massive
MIMO,” CoRR , vol. abs/1801.06619, 2018. [Online]. Available:
 http://arxiv.org/abs/1801.06619 

 

 
 [114] 
 
J. Yoo, H. J. Kim, and K. H. Johansson, “Mapless indoor localization by
trajectory learning from a crowd,” in 2016 International Conference on
Indoor Positioning and Indoor Navigation (IPIN) , 2016, pp. 1–7.

 

 
 [115] 
 
H. Rizk and M. Youssef, “MonoDCell: A ubiquitous and low-overhead deep
learning-based indoor localization with limited cellular information,” in
 Proceedings of the 27th ACM SIGSPATIAL International Conference on
Advances in Geographic Information Systems , ser. SIGSPATIAL ’19. New York, NY, USA: Association for Computing
Machinery, 2019, p. 109–118. [Online]. Available:
 https://doi.org/10.1145/3347146.3359065 

 

 
 [116] 
 
X. Ye, X. Yin, X. Cai, A. Pérez Yuste, and H. Xu,
“Neural-network-assisted UE localization using radio-channel fingerprints
in LTE networks,” IEEE Access , vol. 5, pp. 12 071–12 087, 2017.

 

 
 [117] 
 
Y. Zhang, W. Rao, and Y. Xiao, “Deep neural network-based telco outdoor
localization,” in Proceedings of the 16th ACM Conference on Embedded
Networked Sensor Systems , ser. SenSys ’18. New York, NY, USA: Association for Computing Machinery, 2018, p.
307–308. [Online]. Available: https://doi.org/10.1145/3274783.3275156 

 

 
 [118] 
 
D. Hall, R. Narayanan, and D. Jenkins, “SDR based indoor beacon localization
using 3D probabilistic multipath exploitation and deep learning,”
 Electronics , vol. 8, p. 1323, 11 2019.

 

 
 [119] 
 
Y. J. and K. H. J., “Target localization in wireless sensor networks using
online semi-supervised support vector regression,” Sensors (Basel) ,
2015.

 

 
 [120] 
 
S. Mahfouz, F. Mourad-Chehade, P. Honeine, J. Farah, and H. Snoussi,
“Target tracking using machine learning and kalman filter in wireless sensor
networks,” IEEE Sensors Journal , vol. 14, no. 10, pp. 3715–3725,
2014.

 

 
 [121] 
 
C. Wu, H. Hou, W. Wang, Q. Huang, and X. Gao, “TDOA based indoor positioning
with NLOS identification by machine learning,” 10 2018, pp. 1–6.

 

 
 [122] 
 
Jie Yin, Qiang Yang, and Lionel Ni, “Adaptive temporal radio maps for
indoor location estimation,” in Third IEEE International Conference on
Pervasive Computing and Communications , 2005, pp. 85–94.

 

 
 [123] 
 
X. Zhang, Y. Mei, H. Jin, and D. Liang, “TL-FCMA: Indoor
localization by integrating fuzzy clustering with transfer learning,” in
 2018 International Conference on Network Infrastructure and Digital
Content (IC-NIDC) , 2018, pp. 372–377.

 

 
 [124] 
 
K. Liu, H. Zhang, J. K. Ng, Y. Xia, L. Feng, V. C. S. Lee, and
S. H. Son, “Toward low-overhead fingerprint-based indoor localization via
transfer learning: Design, implementation, and evaluation,” IEEE
Transactions on Industrial Informatics , vol. 14, no. 3, pp. 898–908, 2018.

 

 
 [125] 
 
Y. Li, Z. He, Z. Gao, Y. Zhuang, C. Shi, and N. El-Sheimy, “Toward
robust crowdsourcing-based localization: A fingerprinting accuracy indicator
enhanced wireless/magnetic/inertial integration approach,” IEEE
Internet of Things Journal , vol. 6, no. 2, pp. 3585–3600, 2019.

 

 
 [126] 
 
Y. An Shin, K. Yul Kim, and A. Lan Hong, “Wireless localization method and
wireless localization apparatus using fingerprinting technique,” Nov 2015.

 

 
 [127] 
 
C. Xiao, D. Yang, Z. Chen, and G. Tan, “3-D BLE indoor localization
based on denoising autoencoder,” IEEE Access , vol. 5, pp.
12 751–12 760, 2017.

 

 
 [128] 
 
A. P. Rahmadini, P. Kristalina, and A. Sudarsono, “An improved
fingerprint method based on K-NN algorithm for indoor multiple object
tracking,” in 2017 International Electronics Symposium on Engineering
Technology and Applications (IES-ETA) , 2017, pp. 239–244.

 

 
 [129] 
 
Y. Cheng, R. Y. Chang, and L. Chen, “A comparative study of
machine-learning indoor localization using FM and DVB-T signals in real
testbed environments,” in 2017 IEEE 85th Vehicular Technology
Conference (VTC Spring) , 2017, pp. 1–7.

 

 
 [130] 
 
Y. Shao, L. Li, and X. Guo, “A semi-supervised deep learning approach towards
localization of crowdsourced data,” in Proceedings of the ACM Turing
Celebration Conference - China , ser. ACM TURC ’19. New York, NY, USA: Association for Computing Machinery,
2019. [Online]. Available: https://doi.org/10.1145/3321408.3321584 

 

 
 [131] 
 
H. Zou, H. Wang, L. Xie, and Q. Jia, “An RFID indoor positioning
system by using weighted path loss and extreme learning machine,” in
 2013 IEEE 1st International Conference on Cyber-Physical Systems,
Networks, and Applications (CPSNA) , 2013, pp. 66–71.

 

 
 [132] 
 
J. Zhao, X. Gao, X. Wang, C. Li, M. Song, and Q. Sun, “An efficient radio map
updating algorithm based on K-Means and gaussian process regression,”
 Journal of Navigation , pp. 1–14, 04 2018.

 

 
 [133] 
 
J. Yu, H. M. Saad, and R. M. Buehrer, “Centimeter-level indoor
localization using channel state information with recurrent neural
networks,” in 2020 IEEE/ION Position, Location and Navigation
Symposium (PLANS) , 2020, pp. 1317–1323.

 

 
 [134] 
 
C. Kumar and K. Rajawat, “Dictionary-based statistical fingerprinting for
indoor localization,” IEEE Transactions on Vehicular Technology ,
vol. 68, no. 9, pp. 8827–8841, 2019.

 

 
 [135] 
 
B. Berruet, O. Baala, A. Caminada, and V. Guillet, “E-Loc: Enhanced
CSI fingerprinting localization for massive machine-type communications in
Wi-Fi ambient connectivity,” in 2019 International Conference on
Indoor Positioning and Indoor Navigation (IPIN) , 2019, pp. 1–8.

 

 
 [136] 
 
Z. Chen and J. Wang, “GROF: Indoor localization using a multiple-bandwidth
general regression neural network and outlier filter,” Sensors ,
vol. 18, p. 3723, 11 2018.

 

 
 [137] 
 
J.-S. Leu, M.-C. Yu, and H.-J. Tzeng, “Improving indoor positioning precision
by using received signal strength fingerprint and footprint based on weighted
ambient Wi-Fi signals,” Computer Networks , vol. 91, pp. 329–340,
11 2015.

 

 
 [138] 
 
H. Rizk and M. Youssef, “Increasing coverage of indoor localization systems
for EEE112 support,” 2019.

 

 
 [139] 
 
Q. Zhang, M. Zhou, Z. Tian, and Y. Wang, “Indoor localization using
semi-supervised manifold alignment with dimension expansion,” Applied
Sciences , vol. 6, p. 338, 11 2016.

 

 
 [140] 
 
H. Meng, F. Yuan, T. Yan, and M. Zeng, “Indoor positioning of RBF
neural network based on improved fast clustering algorithm combined with LM
algorithm,” IEEE Access , vol. 7, pp. 5932–5945, 2019.

 

 
 [141] 
 
S. Wang, G. Mao, and J. A. Zhang, “Joint time-of-arrival estimation for
coherent UWB ranging in multipath environment with multi-user
interference,” IEEE Transactions on Signal Processing , vol. 67,
no. 14, pp. 3743–3755, 2019.

 

 
 [142] 
 
P. Chen, J. Shang, and F. Gu, “Learning RSSI feature via ranking model
for Wi-Fi fingerprinting localization,” IEEE Transactions on
Vehicular Technology , vol. 69, no. 2, pp. 1695–1705, 2020.

 

 
 [143] 
 
L. Li, W. Yang, M. Z. A. Bhuiyan, and G. Wang, “Unsupervised learning of
indoor localization based on received signal strength: Unsupervised learning
of indoor localization,” Wireless Communications and Mobile
Computing , vol. 16, 10 2016.

 

 
 [144] 
 
A. Del Corte-Valiente, J. Gómez-Pulido, O. Gutiérrez-Blanco, and
J. Castillo-Sequera, “Localization approach based on ray-tracing simulations
and fingerprinting techniques for indoor–outdoor scenarios,”
 Energies , 07 2019.

 

 
 [145] 
 
P. Rama and S. Murugan, “Localization approach for tracking the mobile nodes
using FA based ANN in subterranean wireless sensor networks,”
 Neural Processing Letters , vol. 51, pp. 1–20, 10 2019.

 

 
 [146] 
 
J. Yoo and K. H. Johansson, “Semi-supervised learning for mobile robot
localization using wireless signal strengths,” in 2017 International
Conference on Indoor Positioning and Indoor Navigation (IPIN) , 2017, pp.
1–8.

 

 
 [147] 
 
W. Qian, F. Lauri, and F. Gechter, “Supervised and semi-supervised deep
learning-based models for indoor location prediction and recognition,” 11
2019.

 

 
 [148] 
 
G. Zhang, P. Wang, H. Chen, and L. Zhang, “Wireless indoor localization using
convolutional neural network and gaussian process regression,”
 Sensors , vol. 19, no. 11, p. 2508, 2019.

 

 
 [149] 
 
L. Wu, C.-H. Chen, and Q. Zhang, “A mobile positioning method based on deep
learning techniques,” Electronics , vol. 8, no. 1, p. 59, 2019.

 

 
 [150] 
 
Z. Feng, Y. Cao, and J. Yan, “A received signal strength based indoor
localization algorithm using ELM technique and ridge regression,” in
 2019 IEEE 2nd International Conference on Electronic Information and
Communication Technology (ICEICT) . IEEE, 2019, pp. 599–603.

 

 
 [151] 
 
R. H. Jaafar and S. S. Saab, “A neural network approach for indoor
fingerprinting-based localization,” in 2018 9th IEEE Annual Ubiquitous
Computing, Electronics Mobile Communication Conference (UEMCON) . IEEE, 2018, pp. 537–542.

 

 
 [152] 
 
G. Félix, M. Siller, and E. N. Alvarez, “A fingerprinting indoor
localization algorithm based deep learning,” in 2016 Eighth
International Conference on Ubiquitous and Future Networks (ICUFN) . IEEE, 2016, pp. 1006–1011.

 

 
 [153] 
 
M. Yan, F. Xu, S. Bai, and Q. Wan, “A noise reduction fingerprint feature for
indoor localization,” in 2018 10th International Conference on
Wireless Communications and Signal Processing (WCSP) . IEEE, 2018, pp. 1–6.

 

 
 [154] 
 
R. Wang, Z. Li, H. Luo, F. Zhao, W. Shao, and Q. Wang, “A robust Wi-Fi
fingerprint positioning algorithm using stacked denoising autoencoder and
multi-layer perceptron,” Remote Sensing , vol. 11, no. 11, p. 1293,
2019.

 

 
 [155] 
 
M. Z. Comiter, M. B. Crouse, and H. Kung, “A data-driven approach to
localization for high frequency wireless mobile networks,” in GLOBECOM
2017-2017 IEEE Global Communications Conference . IEEE, 2017, pp. 1–7.

 

 
 [156] 
 
K. A. Nguyen, “A performance guaranteed indoor positioning system using
conformal prediction and the wifi signal strength,” Journal of
Information and Telecommunication , vol. 1, no. 1, pp. 41–65, 2017.

 

 
 [157] 
 
A. Musa, G. D. Nugraha, H. Han, D. Choi, S. Seo, and J. Kim, “A decision
tree-based NLOS detection method for the uwb indoor location tracking
accuracy improvement,” International Journal of Communication
Systems , vol. 32, no. 13, p. e3997, 2019.

 

 
 [158] 
 
M. Z. Comiter, M. B. Crouse, H. Kung, and J. Paulson, “A structured deep
neural network for data-driven localization in high frequency wireless
networks,” Int. J. Comput. Netw. Commun , vol. 9, no. 3, pp. 21–39,
2017.

 

 
 [159] 
 
H. Zou, X. Lu, H. Jiang, and L. Xie, “A fast and precise indoor localization
algorithm based on an online sequential extreme learning machine,”
 Sensors , vol. 15, no. 1, pp. 1804–1824, 2015.

 

 
 [160] 
 
C. Figuera, J. L. Rojo-Álvarez, M. Wilby, I. Mora-Jiménez, and A. J.
Caamaño, “Advanced support vector machines for 802.11 indoor location,”
 Signal Processing , vol. 92, no. 9, pp. 2126–2136, 2012.

 

 
 [161] 
 
X. Guo, S. Zhu, L. Li, F. Hu, and N. Ansari, “Accurate WiFi localization by
unsupervised fusion of extended candidate location set,” IEEE Internet
of Things Journal , vol. 6, no. 2, pp. 2476–2485, 2018.

 

 
 [162] 
 
J. Bi, Y. Wang, X. Li, H. Qi, H. Cao, and S. Xu, “An adaptive weighted KNN
positioning method based on omnidirectional fingerprint database and twice
affinity propagation clustering,” Sensors , vol. 18, no. 8, p. 2502,
2018.

 

 
 [163] 
 
M. Elbes, E. Almaita, T. Alrawashdeh, T. Kanan, S. AlZu’bi, and B. Hawashin,
“An indoor localization approach based on deep learning for indoor
location-based services,” in 2019 IEEE Jordan International Joint
Conference on Electrical Engineering and Information Technology
(JEEIT) . IEEE, 2019, pp. 437–441.

 

 
 [164] 
 
M. Comiter and H. Kung, “Localization convolutional neural networks using
angle of arrival images,” in 2018 IEEE Global Communications
Conference (GLOBECOM) . IEEE, 2018,
pp. 1–7.

 

 
 [165] 
 
L. Zhang, N. Xiao, J. Li, and W. Yang, “Heterogeneous feature machine learning
for performance-enhancing indoor localization,” in 2018 IEEE 87th
Vehicular Technology Conference (VTC Spring) . IEEE, 2018, pp. 1–5.

 

 
 [166] 
 
A. B. Adege, H.-P. Lin, G. B. Tarekegn, Y. Y. Munaye, and L. Yen, “An indoor
and outdoor positioning using a hybrid of support vector machine and deep
neural network algorithms,” Journal of sensors , vol. 2018, 2018.

 

 
 [167] 
 
A. B. Adege, H.-P. Lin, and L.-C. Wang, “Mobility predictions for IoT
devices using gated recurrent unit network,” IEEE Internet of Things
Journal , 2019.

 

 
 [168] 
 
R. Malik, R. Gustifa, A. Farissi, D. Stiawan, H. Ubaya, M. Ahmad, and
A. Khirbeet, “The indoor positioning system using fingerprint method based
deep neural network,” in IOP Conference Series: Earth and
Environmental Science , vol. 248, no. 1. IOP Publishing, 2019, p. 012077.

 

 
 [169] 
 
H. J. Bae and L. Choi, “Large-scale indoor positioning using geomagnetic field
with deep neural networks,” in ICC 2019-2019 IEEE International
Conference on Communications (ICC) . IEEE, 2019, pp. 1–6.

 

 
 [170] 
 
Z. Yi, J. Zhao, Z. Zhang, and M. Kong, “Neural network based prediction and
analysis for NB-IoT network location,” in 2019 11th International
Conference on Wireless Communications and Signal Processing (WCSP) . IEEE, 2019, pp. 1–5.

 

 
 [171] 
 
T. ZHANG and M. Yi, “The enhancement of WiFi fingerprint positioning using
convolutional neural network,” DEStech Transactions on Computer
Science and Engineering , no. CCNT, 2018.

 

 
 [172] 
 
S. Tewes, A. A. Ahmad, J. Kakar, U. M. Thanthrige, S. Roth, and A. Sezgin,
“Ensemble-based learning in indoor localization: A hybrid approach,” in
 2019 IEEE 90th Vehicular Technology Conference (VTC2019-Fall) . IEEE, 2019, pp. 1–5.

 

 
 [173] 
 
W. Xiao, P. Liu, W.-S. Soh, and G.-B. Huang, “Large scale wireless indoor
localization by clustering and extreme learning machine,” in 2012 15th
International Conference on Information Fusion . IEEE, 2012, pp. 1609–1614.

 

 
 [174] 
 
M. Widmaier, M. Arnold, S. Dorner, S. Cammerer, and S. ten Brink, “Towards
practical indoor positioning based on massive MIMO systems,” in 2019
IEEE 90th Vehicular Technology Conference (VTC2019-Fall) . IEEE, 2019, pp. 1–6.

 

 
 [175] 
 
E. Homayounvala, M. Nabati, R. Shahbazian, S. A. Ghorashi, and V. Moghtadaiee,
“A novel smartphone application for indoor positioning of users based on
machine learning,” in Adjunct Proceedings of the 2019 ACM
International Joint Conference on Pervasive and Ubiquitous Computing and
Proceedings of the 2019 ACM International Symposium on Wearable Computers ,
2019, pp. 430–437.

 

 
 [176] 
 
G. Bhatti, “Machine learning based localization in large-scale wireless sensor
networks,” Sensors , vol. 18, no. 12, p. 4179, 2018.

 

 
 [177] 
 
L. Xiao, A. Behboodi, and R. Mathar, “Learning the localization function:
Machine learning approach to fingerprinting localization,” arXiv
preprint arXiv:1803.08153 , 2018.

 

 
 [178] 
 
S. Mahfouz, F. Mourad-Chehade, P. Honeine, J. Farah, and H. Snoussi,
“Kernel-based machine learning using radio-fingerprints for localization in
WSNs,” IEEE Transactions on Aerospace and Electronic Systems ,
vol. 51, no. 2, pp. 1324–1336, 2015.

 

 
 [179] 
 
C. Qiu and M. W. Mutka, “Walk and learn: Enabling accurate indoor positioning
by profiling outdoor movement on smartphones,” Pervasive and Mobile
Computing , vol. 48, pp. 84–100, 2018.

 

 
 [180] 
 
P. A. Patel, “Millimeter wave positioning with deep learning,” 2020.

 

 
 [181] 
 
S. He, W. Lin, and S. . G. Chan, “Indoor localization and automatic
fingerprint update with altered AP signals,” IEEE Transactions on
Mobile Computing , vol. 16, no. 7, pp. 1897–1910, 2017.

 

 
 [182] 
 
Y. Cheng, H. Chou, and R. Y. Chang, “Machine-learning indoor
localization with access point selection and signal strength
reconstruction,” in 2016 IEEE 83rd Vehicular Technology Conference
(VTC Spring) , 2016, pp. 1–5.

 

 
 [183] 
 
J. Yang, S. Jin, C.-K. Wen, J. Guo, and M. Matthaiou, “3-D positioning and
environment mapping for mmWave communication systems,” 2019.

 

 
 [184] 
 
W. Chen, W. Wang, Q. Li, Q. Chang, and H. Hou, “A crowd-sourcing indoor
localization algorithm via optical camera on a smartphone assisted by Wi-Fi
fingerprint RSSI,” Sensors , vol. 16, p. 410, 03 2016.

 

 
 [185] 
 
S. Liu, Y. Zhao, and B. Chen, “WiCount: A deep learning approach for crowd
counting using WiFi signals,” in 2017 IEEE International Symposium
on Parallel and Distributed Processing with Applications and 2017 IEEE
International Conference on Ubiquitous Computing and Communications
(ISPA/IUCC) . IEEE, 2017, pp.
967–974.

 

 
 [186] 
 
H. Wymeersch, J. Lien, and M. Z. Win, “Cooperative localization in wireless
networks,” Proceedings of the IEEE , vol. 97, no. 2, pp. 427–450,
2009.

 

 
 [187] 
 
A. Ray, S. Deb, and P. Monogioudis, “Localization of LTE measurement records
with missing information,” in IEEE INFOCOM 2016-The 35th Annual IEEE
International Conference on Computer Communications . IEEE, 2016, pp. 1–9.

 

 
 [188] 
 
Y. Zhang, A. Y. Ding, J. Ott, M. Yuan, J. Zeng, K. Zhang, and W. Rao,
“Transfer learning-based outdoor position recovery with telco data,”
 IEEE Transactions on Mobile Computing , 2020.

 

 
 [189] 
 
F. Zhu, C. Luo, M. Yuan, Y. Zhu, Z. Zhang, T. Gu, K. Deng, W. Rao, and J. Zeng,
“City-scale localization with telco big data,” in Proceedings of the
25th ACM International on Conference on Information and Knowledge
Management , 2016, pp. 439–448.

 

 
 [190] 
 
M. M. Butt, A. Rao, and D. Yoon, “RF fingerprinting and deep learning
assisted UE positioning in 5G,” arXiv preprint arXiv:2001.00977 ,
2020.

 

 
 [191] 
 
Y. Zhang, Y. Xiao, K. Zhao, and W. Rao, “DeepLoc: deep neural network-based
telco localization,” in Proceedings of the 16th EAI International
Conference on Mobile and Ubiquitous Systems: Computing, Networking and
Services , 2019, pp. 258–267.

 

 
 [192] 
 
A. Shokry, M. Torki, and M. Youssef, “DeepLoc: a ubiquitous accurate and
low-overhead outdoor cellular localization system,” in Proceedings of
the 26th ACM SIGSPATIAL International Conference on Advances in Geographic
Information Systems , 2018, pp. 339–348.

 

 
 [193] 
 
A. Goswami, L. E. Ortiz, and S. R. Das, “WiGEM: A learning-based approach
for indoor localization,” in Proceedings of the Seventh COnference on
emerging Networking EXperiments and Technologies , 2011, pp. 1–12.

 

 
 [194] 
 
J. Choi, Y.-S. Choi, and S. Talwar, “Unsupervised learning techniques for
trilateration: From theory to android app implementation,” IEEE
Access , vol. 7, pp. 134 525–134 538, 2019.

 

 
 [195] 
 
A. Pandey, R. Vamsi, and S. Kumar, “Handling device heterogeneity and
orientation using multistage regression for GMM based localization in IoT
networks,” IEEE Access , vol. 7, pp. 144 354–144 365, 2019.

 

 
 [196] 
 
B. A. Akram, A. H. Akbar, and K.-H. Kim, “CEnsLoc: infrastructure-less
indoor localization methodology using GMM clustering-based classification
ensembles,” Mobile Information Systems , vol. 2018, 2018.

 

 
 [197] 
 
C.-C. Chui and R. A. Scholtz, “Time transfer in impulse radio networks,”
 IEEE transactions on communications , vol. 57, no. 9, pp. 2771–2781,
2009.

 

 
 [198] 
 
S. Aditya, A. F. Molisch, and H. M. Behairy, “A survey on the impact of
multipath on wideband time-of-arrival based localization,” Proceedings
of the IEEE , vol. 106, no. 7, pp. 1183–1203, 2018.

 

 
 [199] 
 
S. Marano, W. M. Gifford, H. Wymeersch, and M. Z. Win, “NLOS identification
and mitigation for localization based on UWB experimental data,”
 IEEE Journal on selected areas in communications , vol. 28, no. 7, pp.
1026–1035, 2010.

 

 
 [200] 
 
Z. Sahinoglu, Ultra-wideband positioning systems . Cambridge university press, 2008.

 

 
 [201] 
 
S. Wu, S. Zhang, K. Xu, and D. Huang, “Neural network localization with TOA
measurements based on error learning and matching,” IEEE Access ,
vol. 7, pp. 19 089–19 099, 2019.

 

 
 [202] 
 
Y. Xue, W. Su, H. Wang, D. Yang, and Y. Jiang, “DeepTAL: Deep learning for
TDOA-based asynchronous localization security with measurement error and
missing data,” IEEE Access , vol. 7, pp. 122 492–122 502, 2019.

 

 
 [203] 
 
J. Li, I.-T. Lu, J. S. Lu, and L. Zhang, “Robust kernel-based machine learning
localization using NLOS TOAs or TDOAs,” in 2017 IEEE Long Island
Systems, Applications and Technology Conference (LISAT) . IEEE, 2017, pp. 1–6.

 

 
 [204] 
 
M. N. de Sousa, “Enhanced localization systems with multipath fingerprints and
machine learning,” in 2019 IEEE 30th Annual International Symposium on
Personal, Indoor and Mobile Radio Communications (PIMRC) . IEEE, 2019, pp. 1–6.

 

 
 [205] 
 
X. Wang, X. Wang, and S. Mao, “Deep convolutional neural networks for indoor
localization with CSI images,” IEEE Transactions on Network Science
and Engineering , 2018.

 

 
 [206] 
 
Y. Zhang and K. Psounis, “Efficient indoor localization via switched-beam
antennas,” IEEE Transactions on Mobile Computing , 2019.

 

 
 [207] 
 
M. Passafiume, S. Maddio, A. Cidronali, and G. Manes, “MUSIC algorithm for
RSSI-based DoA estimation on standard IEEE 802.11/802.15. x systems.”

 

 
 [208] 
 
X. Wang, L. Liu, Y. Lin, and X. Chen, “A fast single-site fingerprint
localization method in massive MIMO system,” in 2019 11th
International Conference on Wireless Communications and Signal Processing
(WCSP) . IEEE, 2019, pp. 1–6.

 

 
 [209] 
 
X. Sun, X. Gao, G. Y. Li, and W. Han, “Single-site localization based on a new
type of fingerprint for massive MIMO-OFDM systems,” IEEE
Transactions on Vehicular Technology , vol. 67, no. 7, pp. 6134–6145, 2018.

 

 
 [210] 
 
T. Janssen, R. Berkvens, and M. Weyn, “Comparing machine learning algorithms
for RSS-based localization in LPWAN,” in International Conference
on P2P, Parallel, Grid, Cloud and Internet Computing . Springer, 2019, pp. 726–735.

 

 
 [211] 
 
N. Ghourchian, M. Allegue-Martinez, and D. Precup, “Real-time indoor
localization in smart homes using semi-supervised learning,” in
 Twenty-Ninth IAAI Conference , 2017.

 

 
 [212] 
 
C. Wu, X. Yi, W. Wang, L. You, Q. Huang, and X. Gao, “Learning to localize: A
3D CNN approach to user positioning in massive MIMO-OFDM systems,”
 arXiv preprint arXiv:1910.12378 , 2019.

 

 
 [213] 
 
K. S. V. Prasad, E. Hossain, and V. K. Bhargava, “Machine learning methods for
RSS-based user positioning in distributed massive MIMO,” IEEE
Transactions on Wireless Communications , vol. 17, no. 12, pp. 8402–8417,
2018.

 

 
 [214] 
 
X. Sun, C. Wu, X. Gao, and G. Y. Li, “Fingerprint-based localization for
massive MIMO-OFDM system with deep convolutional neural networks,”
 IEEE Transactions on Vehicular Technology , vol. 68, no. 11, pp.
10 846–10 857, 2019.

 

 
 [215] 
 
J. Vieira, E. Leitinger, M. Sarajlic, X. Li, and F. Tufvesson, “Deep
convolutional neural networks for massive MIMO fingerprint-based
positioning,” in 2017 IEEE 28th Annual International Symposium on
Personal, Indoor, and Mobile Radio Communications (PIMRC) . IEEE, 2017, pp. 1–6.

 

 
 [216] 
 
A. Decurninge, L. G. Ordóñez, P. Ferrand, H. Gaoning, L. Bojie, Z. Wei,
and M. Guillaud, “CSI-based outdoor localization for massive MIMO:
Experiments with a learning approach,” in 2018 15th International
Symposium on Wireless Communication Systems (ISWCS) . IEEE, 2018, pp. 1–6.

 

 
 [217] 
 
M. Arnold, S. Dorner, S. Cammerer, and S. Ten Brink, “On deep learning-based
massive MIMO indoor user localization,” in 2018 IEEE 19th
International Workshop on Signal Processing Advances in Wireless
Communications (SPAWC) . IEEE, 2018,
pp. 1–5.

 

 
 [218] 
 
K. S. V. Prasad, E. Hossain, and V. K. Bhargava, “A numerical approximation
method for RSS-based user positioning in distributed massive MIMO,” in
 2017 IEEE International Conference on Advanced Networks and
Telecommunications Systems (ANTS) . IEEE, 2017, pp. 1–6.

 

 
 [219] 
 
H. Pirzadeh, C. Wang, and H. Papadopoulos, “Machine-learning assisted outdoor
localization via sector-based fog massive MIMO,” in ICC 2019-2019
IEEE International Conference on Communications (ICC) . IEEE, 2019, pp. 1–6.

 

 
 [220] 
 
J. Flordelis, X. Li, O. Edfors, and F. Tufvesson, “Massive MIMO extensions
to the COST 2100 channel model: Modeling and validation,” IEEE
Transactions on Wireless Communications , vol. 19, no. 1, pp. 380–394, 2019.

 

 
 [221] 
 
M.-G. Di Benedetto, UWB communication systems: a comprehensive
overview . Hindawi Publishing
Corporation, 2006, vol. 5.

 

 
 [222] 
 
S. Kram, M. Stahlke, T. Feigl, J. Seitz, and J. Thielecke, “UWB channel
impulse responses for positioning in complex environments: A detailed feature
analysis,” Sensors , vol. 19, no. 24, p. 5547, 2019.

 

 
 [223] 
 
S. Krishnan, R. X. M. Santos, E. R. Yap, and M. T. Zin, “Improving UWB based
indoor positioning in industrial environments through machine learning,” in
 2018 15th International Conference on Control, Automation, Robotics and
Vision (ICARCV) . IEEE, 2018, pp.
1484–1488.

 

 
 [224] 
 
H. Wymeersch, S. Maranò, W. M. Gifford, and M. Z. Win, “A machine learning
approach to ranging error mitigation for uwb localization,” IEEE
transactions on communications , vol. 60, no. 6, pp. 1719–1728, 2012.

 

 
 [225] 
 
K. Bregar and M. Mohorčič, “Improving indoor localization using
convolutional neural networks on computationally restricted devices,”
 IEEE Access , vol. 6, pp. 17 429–17 441, 2018.

 

 
 [226] 
 
B. Berruet, O. Baala, A. Caminada, and V. Guillet, “DelFin: a deep learning
based CSI fingerprinting indoor localization in IoT context,” in
 2018 International Conference on Indoor Positioning and Indoor
Navigation (IPIN) . IEEE, 2018, pp.
1–8.

 

 
 [227] 
 
C.-M. Own, J. Hou, and W. Tao, “Signal fuse learning method with dual bands
WiFi signal measurements in indoor positioning,” IEEE Access ,
vol. 7, pp. 131 805–131 817, 2019.

 

 
 [228] 
 
X. Wang, L. Gao, and S. Mao, “CSI phase fingerprinting for indoor
localization with a deep learning approach,” IEEE Internet of Things
Journal , vol. 3, no. 6, pp. 1113–1123, 2016.

 

 
 [229] 
 
K. Wu, M. Yang, C. Ma, and J. Yan, “CSI-based wireless localization and
activity recognition using support vector machine,” in 2019 IEEE
International Conference on Signal Processing, Communications and Computing
(ICSPCC) . IEEE, 2019, pp. 1–5.

 

 
 [230] 
 
T. Li, H. Wang, Y. Shao, and Q. Niu, “Channel state information–based
multi-level fingerprinting for indoor localization with deep learning,”
 International Journal of Distributed Sensor Networks , vol. 14, no. 10,
p. 1550147718806719, 2018.

 

 
 [231] 
 
X. Wang, X. Wang, and S. Mao, “CiFi: Deep convolutional neural networks for
indoor localization with 5 GHz Wi-Fi,” in 2017 IEEE International
Conference on Communications (ICC) . IEEE, 2017, pp. 1–6.

 

 
 [232] 
 
H. Li, X. Zeng, Y. Li, S. Zhou, and J. Wang, “Convolutional neural networks
based indoor Wi-Fi localization with a novel kind of CSI images,”
 China Communications , vol. 16, no. 9, pp. 250–260, 2019.

 

 
 [233] 
 
C.-H. Hsieh, J.-Y. Chen, and B.-H. Nien, “Deep learning-based indoor
localization using received signal strength and channel state information,”
 IEEE access , vol. 7, pp. 33 256–33 267, 2019.

 

 
 [234] 
 
X. Wang, L. Gao, S. Mao, and S. Pandey, “DeepFi: Deep learning for indoor
fingerprinting using channel state information,” in 2015 IEEE wireless
communications and networking conference (WCNC) . IEEE, 2015, pp. 1666–1671.

 

 
 [235] 
 
Z. Wu, Q. Xu, J. Li, C. Fu, Q. Xuan, and Y. Xiang, “Passive indoor
localization based on CSI and naive Bayes classification,” IEEE
Transactions on Systems, Man, and Cybernetics: Systems , vol. 48, no. 9, pp.
1566–1577, 2017.

 

 
 [236] 
 
X. Wang, X. Wang, and S. Mao, “ResLoc: Deep residual sharing learning for
indoor localization with CSI tensors,” in 2017 IEEE 28th Annual
International Symposium on Personal, Indoor, and Mobile Radio Communications
(PIMRC) . IEEE, 2017, pp. 1–6.

 

 
 [237] 
 
T. Fukushima, T. Murakami, H. Abeysekera, S. Saruwatari, and T. Watanabe,
“Evaluating indoor localization performance on an IEEE 802.11 ac
explicit-feedback-based CSI learning system,” in 2019 IEEE 89th
Vehicular Technology Conference (VTC2019-Spring) . IEEE, 2019, pp. 1–6.

 

 
 [238] 
 
A. Sobehy, É. Renault, and P. Mühlethaler, “CSI based indoor
localization using ensemble neural networks,” in International
Conference on Machine Learning for Networking . Springer, 2019, pp. 367–378.

 

 
 [239] 
 
E. Lei, O. Castañeda, O. Tirkkonen, T. Goldstein, and C. Studer, “Siamese
neural networks for wireless positioning and channel charting,” in
 2019 57th Annual Allerton Conference on Communication, Control, and
Computing (Allerton) . IEEE, 2019, pp.
200–207.

 

 
 [240] 
 
A. Niitsoo, T. Edelhäußer, E. Eberlein, N. Hadaschik, and C. Mutschler,
“A deep learning approach to position estimation from channel impulse
responses,” Sensors , vol. 19, no. 5, p. 1064, 2019.

 

 
 [241] 
 
S. Chen, J. Fan, X. Luo, and Y. Zhang, “Multipath-based CSI fingerprinting
localization with a machine learning approach,” in 2018 Wireless
Advanced (WiAd) . IEEE, 2018, pp.
1–5.

 

 
 [242] 
 
Y. Jing, J. Hao, and P. Li, “Learning spatiotemporal features of CSI for
indoor localization with dual-stream 3D convolutional neural networks,”
 IEEE Access , vol. 7, pp. 147 571–147 585, 2019.

 

 
 [243] 
 
S. D. Bast, A. P. Guevara, and S. Pollin, “CSI-based positioning in
massive MIMO systems using convolutional neural networks,” in 2020
IEEE 91st Vehicular Technology Conference (VTC2020-Spring) , 2020, pp. 1–5.

 

 
 [244] 
 
V. Savic, E. G. Larsson, J. Ferrer-Coll, and P. Stenumgaard, “Kernel
methods for accurate UWB-based ranging with reduced complexity,”
 IEEE Transactions on Wireless Communications , vol. 15, no. 3, pp.
1783–1793, 2016.

 

 
 [245] 
 
R. Faragher and R. Harle, “Location fingerprinting with bluetooth low energy
beacons,” IEEE journal on Selected Areas in Communications , vol. 33,
no. 11, pp. 2418–2428, 2015.

 

 
 [246] 
 
R. Margolies, R. Becker, S. Byers, S. Deb, R. Jana, S. Urbanek, and
C. Volinsky, “Can you find me now? Evaluation of network-based
localization in a 4G LTE network,” in IEEE INFOCOM 2017-IEEE
Conference on Computer Communications . IEEE, 2017, pp. 1–9.

 

 
 [247] 
 
J.-Y. Lee, C. Eom, Y. Kwak, H.-G. Kang, and C. Lee, “DNN-based wireless
positioning in an outdoor environment,” in 2018 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP) . IEEE, 2018, pp. 3799–3803.

 

 
 [248] 
 
J. Zhang, P. V. Orlik, Z. Sahinoglu, A. F. Molisch, and P. Kinney, “UWB
systems for wireless sensor networks,” Proceedings of the IEEE ,
vol. 97, no. 2, pp. 313–331, 2009.

 

 
 [249] 
 
L. Zhang, Y. Li, Y. Gu, and W. Yang, “An efficient machine learning approach
for indoor localization,” China Communications , vol. 14, no. 11, pp.
141–150, 2017.

 

 
 [250] 
 
H. Ge, F. Jiang, and Z. Zhang, “A hybrid localization algorithm of rss and toa
based on an ensembled neural network,” in 2019 IEEE 8th Joint
International Information Technology and Artificial Intelligence Conference
(ITAIC) . IEEE, 2019, pp. 1280–1284.

 

 
 [251] 
 
Z. Sun, Y. Chen, J. Qi, and J. Liu, “Adaptive localization through transfer
learning in indoor wi-fi environment,” in 2008 Seventh International
Conference on Machine Learning and Applications . IEEE, 2008, pp. 331–336.

 

 
 [252] 
 
J. Mei and J. Xi, “A novel indoor localization scheme based on refined
fingerprint-based autoencoder network,” in MATEC Web of Conferences ,
vol. 189. EDP Sciences, 2018, p.
03017.

 

 
 [253] 
 
C. Sung and D. Han, “Neural network for predicting error of ap location
estimation method using crowdsourced wi-fi fingerprints,” in 2019 20th
IEEE International Conference on Mobile Data Management (MDM) . IEEE, 2019, pp. 420–424.

 

 
 [254] 
 
M. Dashti and H. Claussen, “Extracting location information from rf
fingerprints,” in 2016 IEEE Globecom Workshops (GC Wkshps) . IEEE, 2016, pp. 1–6.

 

 
 [255] 
 
W. Qian, F. Lauri, and F. Gechter, “Convolutional mixture density recurrent
neural network for predicting user location with WiFi fingerprints,”
 arXiv preprint arXiv:1911.09344 , 2019.

 

 
 [256] 
 
M. Ibrahim, M. Torki, and M. ElNainay, “CNN based indoor localization using
RSS time-series,” in 2018 IEEE Symposium on Computers and
Communications (ISCC) . IEEE, 2018,
pp. 01 044–01 049.

 

 
 [257] 
 
Z. Turgut, S. Üstebay, G. Z. G. Aydın, and A. Sertbaş, “Deep
learning in indoor localization using WiFi,” in International
Telecommunications Conference . Springer, 2019, pp. 101–110.

 

 
 [258] 
 
X. Gan, B. Yu, L. Huang, and Y. Li, “Deep learning for weights training and
indoor positioning using multi-sensor fingerprint,” in 2017
International Conference on Indoor Positioning and Indoor Navigation
(IPIN) . IEEE, 2017, pp. 1–7.

 

 
 [259] 
 
J. Zou, X. Guo, L. Li, S. Zhu, and X. Feng, “Deep regression model for
received signal strength based WiFi localization,” in 2018 IEEE 23rd
International Conference on Digital Signal Processing (DSP) . IEEE, 2018, pp. 1–4.

 

 
 [260] 
 
W. Zhang, K. Liu, W. Zhang, Y. Zhang, and J. Gu, “Deep neural networks for
wireless localization in indoor and outdoor environments,”
 Neurocomputing , vol. 194, pp. 279–287, 2016.

 

 
 [261] 
 
M. Zhou, Y. Tang, Z. Tian, L. Xie, and W. Nie, “Robust neighborhood graphing
for semi-supervised indoor localization with light-loaded location
fingerprinting,” IEEE Internet of Things Journal , vol. 5, no. 5, pp.
3378–3387, 2017.

 

 
 [262] 
 
S. Sorour, Y. Lostanlen, S. Valaee, and K. Majeed, “Joint indoor localization
and radio map construction with limited deployment load,” IEEE
Transactions on Mobile Computing , vol. 14, no. 5, pp. 1031–1043, 2014.

 

 
 [263] 
 
R. W. Ouyang, A. K.-S. Wong, C.-T. Lea, and M. Chiang, “Indoor location
estimation with reduced calibration exploiting unlabeled data via hybrid
generative/discriminative learning,” IEEE transactions on mobile
computing , vol. 11, no. 11, pp. 1613–1626, 2011.

 

 
 [264] 
 
S. Liu, H. Luo, and S. Zou, “A low-cost and accurate indoor localization
algorithm using label propagation based semi-supervised learning,” in
 2009 Fifth International Conference on Mobile Ad-hoc and Sensor
Networks . IEEE, 2009, pp. 108–111.

 

 
 [265] 
 
Y. Gu, Y. Chen, J. Liu, and X. Jiang, “Semi-supervised deep extreme learning
machine for Wi-Fi based localization,” Neurocomputing , vol. 166,
pp. 282–293, 2015.

 

 
 [266] 
 
T. Pulkkinen, T. Roos, and P. Myllymäki, “Semi-supervised learning for
WLAN positioning,” in International Conference on Artificial Neural
Networks . Springer, 2011, pp.
355–362.

 

 
 [267] 
 
M. Zhou, Y. Tang, Z. Tian, and X. Geng, “Semi-supervised learning for indoor
hybrid fingerprint database calibration with low effort,” IEEE
Access , vol. 5, pp. 4388–4400, 2017.

 

 
 [268] 
 
Y. Wang, J. Gao, Z. Li, and L. Zhao, “Robust and accurate Wi-Fi fingerprint
location recognition method based on deep neural network,” Applied
Sciences , vol. 10, no. 1, p. 321, 2020.

 

 
 [269] 
 
S. Aikawa, S. Yamamoto, and M. Morimoto, “WLAN finger print localization
using deep learning,” in 2018 IEEE Asia-Pacific Conference on Antennas
and Propagation (APCAP) . IEEE, 2018,
pp. 541–542.

 

 
 [270] 
 
M. Abbas, M. Elhamshary, H. Rizk, M. Torki, and M. Youssef, “WiDeep:
WiFi-based accurate and robust indoor localization system using deep
learning,” in 2019 IEEE International Conference on Pervasive
Computing and Communications (PerCom . IEEE, 2019, pp. 1–10.

 

 
 [271] 
 
H. Huang and S. Lin, “WiDet: Wi-Fi based device-free passive person
detection with deep convolutional neural networks,” Computer
Communications , vol. 150, pp. 357–366, 2020.

 

 
 [272] 
 
Z. Chen, H. Zou, J. Yang, H. Jiang, and L. Xie, “WiFi
fingerprinting indoor localization using local feature-based deep LSTM,”
 IEEE Systems Journal , vol. 14, no. 2, pp. 3001–3010, 2020.

 

 
 [273] 
 
B. S. Ciftler, A. Albaseer, N. Lasla, and M. Abdallah, “Federated learning for
localization: A privacy-preserving crowdsourcing method,” arXiv
preprint arXiv:2001.01911 , 2020.

 

 
 [274] 
 
M. Zhou, Y. Tang, W. Nie, L. Xie, and X. Yang, “GrassMA: Graph-based
semi-supervised manifold alignment for indoor WLAN localization,”
 IEEE Sensors Journal , vol. 17, no. 21, pp. 7086–7095, 2017.

 

 
 [275] 
 
M. Zhou, Q. Zhang, Z. Tian, and Y. Wang, “Indoor WLAN localization using
high-dimensional manifold alignment with limited calibration load,” in
 2017 IEEE International Conference on Communications (ICC) . IEEE, 2017, pp. 1–6.

 

 
 [276] 
 
S. Chen, Q. Zhu, Z. Li, and Y. Long, “Deep neural network based on feature
fusion for indoor wireless localization,” in 2018 International
Conference on Microwave and Millimeter Wave Technology (ICMMT) . IEEE, 2018, pp. 1–3.

 

 
 [277] 
 
S.-h. Jung, B.-c. Moon, and D. Han, “Unsupervised learning for crowdsourced
indoor localization in wireless networks,” IEEE Transactions on Mobile
Computing , vol. 15, no. 11, pp. 2892–2906, 2015.

 

 
 [278] 
 
H. Wang, S. Sen, A. Elgohary, M. Farid, M. Youssef, and R. R. Choudhury, “No
need to war-drive: Unsupervised indoor localization,” in Proceedings
of the 10th international conference on Mobile systems, applications, and
services , 2012, pp. 197–210.

 

 
 [279] 
 
C. Wu, Z. Yang, Y. Liu, and W. Xi, “WILL: Wireless indoor localization
without site survey,” IEEE Transactions on Parallel and Distributed
Systems , vol. 24, no. 4, pp. 839–848, 2012.

 

 
 [280] 
 
H. Rizk, “Device-invariant cellular-based indoor localization system using
deep learning,” in The ACM MobiSys 2019 on Rising Stars Forum , 2019,
pp. 19–23.

 

 
 [281] 
 
J.-W. Jang and S.-N. Hong, “Indoor localization with WiFi fingerprinting
using convolutional neural network,” in 2018 Tenth International
Conference on Ubiquitous and Future Networks (ICUFN) . IEEE, 2018, pp. 753–758.

 

 
 [282] 
 
H. Rizk, M. Torki, and M. Youssef, “CellinDeep: Robust and accurate
cellular-based indoor localization via deep learning,” IEEE Sensors
Journal , vol. 19, no. 6, pp. 2305–2312, 2018.

 

 
 [283] 
 
H. Rizk, “SoloCell: Efficient indoor localization based on limited cell
network information and minimal fingerprinting,” in Proceedings of the
27th ACM SIGSPATIAL International Conference on Advances in Geographic
Information Systems , 2019, pp. 604–605.

 

 
 [284] 
 
B. Xu, X. Zhu, and H. Zhu, “An efficient indoor localization method based on
the long short-term memory recurrent neuron network,” IEEE Access ,
vol. 7, pp. 123 912–123 921, 2019.

 

 
 [285] 
 
Y. Li, X. Hu, Y. Zhuang, Z. Gao, P. Zhang, and N. El-Sheimy, “Deep
reinforcement learning (DRL): Another perspective for unsupervised wireless
localization,” IEEE Internet of Things Journal , 2019.

 

 
 [286] 
 
T. Ebuchi and H. Yamamoto, “Vehicle/pedestrian localization system using
multiple radio beacons and machine learning for smart parking,” in
 2019 International Conference on Artificial Intelligence in Information
and Communication (ICAIIC) . IEEE,
2019, pp. 086–091.

 

 
 [287] 
 
Y. Peng, W. Fan, X. Dong, and X. Zhang, “An iterative weighted KNN (IW-KNN)
based indoor localization method in bluetooth low energy (BLE)
environment,” in 2016 Intl IEEE Conferences on Ubiquitous Intelligence
 Computing, Advanced and Trusted Computing, Scalable Computing and
Communications, Cloud and Big Data Computing, Internet of People, and Smart
World Congress (UIC/ATC/ScalCom/CBDCom/IoP/SmartWorld) . IEEE, 2016, pp. 794–800.

 

 
 [288] 
 
Q. Li, K. Zhang, M. Cheffena, and X. S. Shen, “A measurement-based boundary
estimation approach for localization in industrial WSNs,” in 2017
IEEE International Conference on Communications (ICC) . IEEE, 2017, pp. 1–6.

 

 
 [289] 
 
J. Chen, C. Wang, Y. Sun, and X. S. Shen, “Semi-supervised Laplacian
regularized least squares algorithm for localization in wireless sensor
networks,” Computer Networks , vol. 55, no. 10, pp. 2481–2491, 2011.

 

 
 [290] 
 
H. Ahmadi and R. Bouallegue, “RSSI-based localization in wireless sensor
networks using regression tree,” in 2015 International Wireless
Communications and Mobile Computing Conference (IWCMC) . IEEE, 2015, pp. 1548–1553.

 

 
 [291] 
 
S. BelMannoubi, H. Touati, and H. Snoussi, “Stacked auto-encoder for scalable
indoor localization in wireless sensor networks,” in 2019 15th
International Wireless Communications Mobile Computing Conference
(IWCMC) . IEEE, 2019, pp. 1245–1250.

 

 
 [292] 
 
S. Aikawa, S. Yamamoto, and T. Muramatsu, “CNN localization using AP
inverse position estimation,” in 2019 IEEE Conference on Antenna
Measurements Applications (CAMA) . IEEE, 2019, pp. 1–3.

 

 
 [293] 
 
V. W. Zheng, H. Cao, S. Gao, A. Adhikari, M. Lin, and K. C.-C. Chang,
“Cold-start heterogeneous-device wireless localization,” in Thirtieth
AAAI Conference on Artificial Intelligence , 2016.

 

 
 [294] 
 
W. Njima, I. Ahriz, R. Zayani, M. Terre, and R. Bouallegue, “Deep CNN for
indoor localization in IoT-sensor systems,” Sensors , vol. 19,
no. 14, p. 3127, 2019.

 

 
 [295] 
 
J. Wang, X. Zhang, Q. Gao, H. Yue, and H. Wang, “Device-free wireless
localization and activity recognition: A deep learning approach,” IEEE
Transactions on Vehicular Technology , vol. 66, no. 7, pp. 6258–6267, 2016.

 

 
 [296] 
 
M. I. AlHajri, N. T. Ali, and R. M. Shubair, “Indoor localization for IoT
using adaptive feature selection: a cascaded machine learning approach,”
 IEEE Antennas and Wireless Propagation Letters , vol. 18, no. 11, pp.
2306–2310, 2019.

 

 
 [297] 
 
X. Wang, L. Gao, S. Mao, and S. Pandey, “CSI-based fingerprinting for indoor
localization: A deep learning approach,” IEEE Transactions on
Vehicular Technology , vol. 66, no. 1, pp. 763–776, 2016.

 

 
 [298] 
 
P. Yazdanian and V. Pourahmadi, “DeepPos: Deep supervised autoencoder
network for CSI based indoor localization,” arXiv preprint
arXiv:1811.12182 , 2018.

 

 
 [299] 
 
R. Zhou, M. Hao, X. Lu, M. Tang, and Y. Fu, “Device-free localization based on
CSI fingerprints and deep neural networks,” in 2018 15th Annual IEEE
International Conference on Sensing, Communication, and Networking
(SECON) . IEEE, 2018, pp. 1–9.

 

 
 [300] 
 
S. Sen, B. Radunovic, R. R. Choudhury, and T. Minka, “You are facing the Mona
Lisa: Spot localization using PHY layer information,” in
 Proceedings of the 10th international conference on Mobile systems,
applications, and services , 2012, pp. 183–196.

 

 
 [301] 
 
A. Niitsoo, T. Edelhäu β \beta er, and C. Mutschler, “Convolutional neural
networks for position estimation in TDoA-based locating systems,” in
 2018 International Conference on Indoor Positioning and Indoor
Navigation (IPIN) . IEEE, 2018, pp.
1–8.

 

 
 [302] 
 
J. Choi, Y.-S. Choi, and S. Talwar, “Unsupervised learning technique to obtain
the coordinates of Wi-Fi access points,” in 2019 International
Conference on Indoor Positioning and Indoor Navigation (IPIN) . IEEE, 2019, pp. 1–6.

 

 
 [303] 
 
M. Roshanaei and M. Maleki, “Dynamic-KNN: A novel locating method in WLAN
based on angle of arrival,” in 2009 IEEE Symposium on Industrial
Electronics Applications , vol. 2. IEEE, 2009, pp. 722–726.

 

 
 [304] 
 
P. Newson and J. Krumm, “Hidden markov map matching through noise and
sparseness,” in Proceedings of the 17th ACM SIGSPATIAL international
conference on advances in geographic information systems , 2009, pp.
336–343.

 

 
 [305] 
 
W. Zhang, R. Sengupta, J. Fodero, and X. Li, “DeepPositioning: Intelligent
fusion of pervasive magnetic field and wifi fingerprinting for smartphone
indoor localization via deep learning,” in 2017 16th IEEE
International Conference on Machine Learning and Applications (ICMLA) . IEEE, 2017, pp. 7–13.

 

 
 [306] 
 
W. Shao, H. Luo, F. Zhao, Y. Ma, Z. Zhao, and A. Crivello, “Indoor positioning
based on fingerprint-image and deep learning,” IEEE Access , vol. 6,
pp. 74 699–74 712, 2018.

 

 
 [307] 
 
B. Soro and C. Lee, “Joint time-frequency RSSI features for convolutional
neural network-based indoor fingerprinting localization,” IEEE
Access , vol. 7, pp. 104 892–104 899, 2019.

 

 
 [308] 
 
A. Kushki, K. N. Plataniotis, and A. N. Venetsanopoulos, “Kernel-based
positioning in wireless local area networks,” IEEE transactions on
mobile computing , vol. 6, no. 6, pp. 689–705, 2007.

 

 
 [309] 
 
Y. Chen, Q. Yang, J. Yin, and X. Chai, “Power-efficient access-point selection
for indoor location estimation,” IEEE Transactions on Knowledge and
Data Engineering , vol. 18, no. 7, pp. 877–888, 2006.

 

 
 [310] 
 
L. Lacasa, B. Luque, F. Ballesteros, J. Luque, and J. C. Nuno, “From time
series to complex networks: The visibility graph,” Proceedings of the
National Academy of Sciences , vol. 105, no. 13, pp. 4972–4975, 2008.

 

 
 [311] 
 
M. R. Gholami, H. Wymeersch, E. G. Ström, and M. Rydström, “Robust
distributed positioning algorithms for cooperative networks,” in 2011
IEEE 12th International Workshop on Signal Processing Advances in Wireless
Communications . IEEE, 2011, pp.
156–160.

 

 
 [312] 
 
S. Aditya, A. F. Molisch, N. Rabeah, and H. M. Behairy, “Localization of
multiple targets with identical radar signatures in multipath environments
with correlated blocking,” IEEE Transactions on Wireless
Communications , vol. 17, no. 1, pp. 606–618, 2017.

 

 
 [313] 
 
A. Conti, S. Mazuelas, S. Bartoletti, W. C. Lindsey, and M. Z. Win, “Soft
information for localization-of-things,” Proceedings of the IEEE ,
vol. 107, no. 11, pp. 2240–2264, 2019.

 

 
 [314] 
 
V. Savic, E. G. Larsson, J. Ferrer-Coll, and P. Stenumgaard, “Kernel principal
component analysis for UWB-based ranging,” in 2014 IEEE 15th
International Workshop on Signal Processing Advances in Wireless
Communications (SPAWC) . IEEE, 2014,
pp. 145–149.

 

 
 [315] 
 
P. Bahl and V. N. Padmanabhan, “Radar: An in-building RF-based user location
and tracking system,” in Proceedings IEEE INFOCOM 2000. Conference on
Computer Communications. Nineteenth Annual Joint Conference of the IEEE
Computer and Communications Societies (Cat. No. 00CH37064) , vol. 2. Ieee, 2000, pp. 775–784.

 

 
 [316] 
 
A. Mittal, S. Tiku, and S. Pasricha, “Adapting convolutional neural networks
for indoor localization with smart mobile devices,” in Proceedings of
the 2018 on Great Lakes Symposium on VLSI , 2018, pp. 117–122.

 

 
 [317] 
 
J. Wang, A. Hu, C. Liu, and X. Li, “A floor-map-aided WiFi/pseudo-odometry
integration algorithm for an indoor positioning system,” Sensors ,
vol. 15, no. 4, pp. 7096–7124, 2015.

 

 
 [318] 
 
J. Ma, X. Li, X. Tao, and J. Lu, “Cluster filtered KNN: A WLAN-based
indoor positioning scheme,” in 2008 International Symposium on a World
of Wireless, Mobile and Multimedia Networks . IEEE, 2008, pp. 1–8.

 

 
 [319] 
 
P. Torteeka and X. Chundi, “Indoor positioning based on Wi-Fi fingerprint
technique using fuzzy K-nearest neighbor,” in Proceedings of 2014
11th International Bhurban Conference on Applied Sciences Technology
(IBCAST) Islamabad, Pakistan, 14th-18th January, 2014 . IEEE, 2014, pp. 461–465.

 

 
 [320] 
 
A. W. Tsui, Y.-H. Chuang, and H.-H. Chu, “Unsupervised learning for solving
RSS hardware variance problem in WiFi localization,” Mobile
Networks and Applications , vol. 14, no. 5, pp. 677–691, 2009.

 

 
 [321] 
 
K. Zhu, “Machine learning in indoor positioning and channel prediction
systems,” 2010.

 

 
 [322] 
 
B. Shin, J. H. Lee, T. Lee, and H. S. Kim, “Enhanced weighted K-nearest
neighbor algorithm for indoor Wi-Fi positioning systems,” in 2012
8th International Conference on Computing Technology and Information
Management (NCM and ICNIT) , vol. 2. IEEE, 2012, pp. 574–577.

 

 
 [323] 
 
B. Altintas and T. Serif, “Improving RSS-based indoor positioning algorithm
via k-means clustering,” in 17th European Wireless 2011-Sustainable
Wireless Technologies . VDE, 2011, pp.
1–5.

 

 
 [324] 
 
J. J. Pan, J. T. Kwok, Q. Yang, and Y. Chen, “Accurate and low-cost location
estimation using kernels.”

 

 
 [325] 
 
Z.-l. Wu, C.-h. Li, J. K.-Y. Ng, and K. R. Leung, “Location estimation via
support vector regression,” IEEE Transactions on mobile computing ,
vol. 6, no. 3, pp. 311–321, 2007.

 

 
 [326] 
 
A. Bekkali, T. Masuo, T. Tominaga, N. Nakamoto, and H. Ban, “Gaussian
processes for learning-based indoor localization,” in 2011 IEEE
International Conference on Signal Processing, Communications and Computing
(ICSPCC) . IEEE, 2011, pp. 1–6.

 

 
 [327] 
 
C. E. Rasmussen, “Gaussian processes in machine learning,” in Summer
School on Machine Learning . Springer,
2003, pp. 63–71.

 

 
 [328] 
 
S. Kumar, R. M. Hegde, and N. Trigoni, “Gaussian process regression for
fingerprinting based localization,” Ad Hoc Networks , vol. 51, pp.
1–10, 2016.

 

 
 [329] 
 
M. M. Atia, A. Noureldin, and M. J. Korenberg, “Dynamic online-calibrated
radio maps for indoor positioning in wireless local area networks,”
 IEEE Transactions on Mobile Computing , vol. 12, no. 9, pp. 1774–1787,
2012.

 

 
 [330] 
 
N. Bui, M. Cesana, S. A. Hosseini, Q. Liao, I. Malanchini, and J. Widmer, “A
survey of anticipatory mobile networking: Context-based classification,
prediction methodologies, and optimization techniques,” IEEE
Communications Surveys Tutorials , vol. 19, no. 3, pp. 1790–1821, 2017.

 

 
 [331] 
 
X. Wang, L. Gao, and S. Mao, “PhaseFi: Phase fingerprinting for indoor
localization with a deep learning approach,” in 2015 IEEE Global
Communications Conference (GLOBECOM) . IEEE, 2015, pp. 1–6.

 

 
 [332] 
 
S. Sabour, N. Frosst, and G. E. Hinton, “Dynamic routing between capsules,”
in Advances in neural information processing systems , 2017, pp.
3856–3866.

 

 
 [333] 
 
H. Chen, Y. Zhang, W. Li, X. Tao, and P. Zhang, “ConFi: Convolutional neural
networks based indoor Wi-Fi localization using channel state information,”
 IEEE Access , vol. 5, pp. 18 066–18 074, 2017.

 

 
 [334] 
 
H. Turabieh and A. Sheta, “Cascaded layered recurrent neural network for
indoor localization in wireless sensor networks,” in 2019 2nd
International Conference on new Trends in Computing Sciences (ICTCS) . IEEE, 2019, pp. 1–6.

 

 
 [335] 
 
G. B. Tarekegn, H.-P. Lin, A. B. Adege, Y. Y. Munaye, and S.-S. Jeng,
“Applying long short-term memory (LSTM) mechanisms for fingerprinting
outdoor positioning in hybrid networks,” in 2019 IEEE 90th Vehicular
Technology Conference (VTC2019-Fall) . IEEE, 2019, pp. 1–5.

 

 
 [336] 
 
X. Song, X. Fan, C. Xiang, Q. Ye, L. Liu, Z. Wang, X. He, N. Yang, and G. Fang,
“A novel convolutional neural network based indoor localization framework
with WiFi fingerprinting,” IEEE Access , vol. 7, pp.
110 698–110 709, 2019.

 

 
 [337] 
 
G. E. Hinton and R. R. Salakhutdinov, “Reducing the dimensionality of data
with neural networks,” science , vol. 313, no. 5786, pp. 504–507,
2006.

 

 
 [338] 
 
O. Chapelle, B. Scholkopf, and A. Zien, “Semi-supervised learning (chapelle,
o. et al., eds.; 2006)[book reviews],” IEEE Transactions on Neural
Networks , vol. 20, no. 3, pp. 542–542, 2009.

 

 
 [339] 
 
Y. Ma and Y. Fu, Manifold learning theory and applications . CRC press, 2011.

 

 
 [340] 
 
V. Pourahmadi and S. Valaee, “Indoor positioning and distance-aware
graph-based semi-supervised learning method,” in 2012 IEEE Global
Communications Conference (GLOBECOM) . IEEE, 2012, pp. 315–320.

 

 
 [341] 
 
J. Yoo, “Time-series Laplacian semi-supervised learning for indoor
localization,” Sensors , vol. 19, no. 18, p. 3867, 2019.

 

 
 [342] 
 
M. Belkin, P. Niyogi, and V. Sindhwani, “Manifold regularization: A geometric
framework for learning from labeled and unlabeled examples,” Journal
of machine learning research , vol. 7, no. Nov, pp. 2399–2434, 2006.

 

 
 [343] 
 
O. Chapelle, V. Vapnik, and J. Weston, “Transductive inference for estimating
values of functions,” in Advances in Neural Information Processing
Systems , 2000, pp. 421–427.

 

 
 [344] 
 
J. Liu, Y. Chen, M. Liu, and Z. Zhao, “SELM: Semi-supervised ELM with
application in sparse calibrated location estimation,”
 Neurocomputing , vol. 74, no. 16, pp. 2566–2572, 2011.

 

 
 [345] 
 
X. Jiang, Y. Chen, J. Liu, Y. Gu, and L. Hu, “FSELM: fusion semi-supervised
extreme learning machine for indoor localization with Wi-Fi and bluetooth
fingerprints,” Soft Computing , vol. 22, no. 11, pp. 3621–3635, 2018.

 

 
 [346] 
 
D. P. Kingma, S. Mohamed, D. J. Rezende, and M. Welling, “Semi-supervised
learning with deep generative models,” in Advances in neural
information processing systems , 2014, pp. 3581–3589.

 

 
 [347] 
 
B. Chidlovskii and L. Antsfeld, “Semi-supervised variational autoencoder for
WiFi indoor localization,” in 2019 International Conference on
Indoor Positioning and Indoor Navigation (IPIN) . IEEE, 2019, pp. 1–8.

 

 
 [348] 
 
D. V. Le, N. Meratnia, and P. J. Havinga, “Unsupervised deep feature learning
to reduce the collection of fingerprints for indoor localization using deep
belief networks,” in 2018 International Conference on Indoor
Positioning and Indoor Navigation (IPIN) . IEEE, 2018, pp. 1–7.

 

 
 [349] 
 
J.-g. Park, B. Charrow, D. Curtis, J. Battat, E. Minkov, J. Hicks, S. Teller,
and J. Ledlie, “Growing an organic indoor location system,” in
 Proceedings of the 8th international conference on Mobile systems,
applications, and services , 2010, pp. 271–284.

 

 
 [350] 
 
Y. Shang, W. Ruml, Y. Zhang, and M. P. Fromherz, “Localization from mere
connectivity,” in Proceedings of the 4th ACM international symposium
on Mobile ad hoc networking computing , 2003, pp. 201–212.

 

 
 [351] 
 
Z. Gao, Y. Ma, K. Liu, X. Miao, and Y. Zhao, “An indoor multi-tag cooperative
localization algorithm based on NMDS for RFID,” IEEE Sensors
Journal , vol. 17, no. 7, pp. 2120–2128, 2017.

 

 
 [352] 
 
N. Saeed and H. Nam, “Robust multidimensional scaling for cognitive radio
network localization,” IEEE Transactions on Vehicular Technology ,
vol. 64, no. 9, pp. 4056–4062, 2014.

 

 
 [353] 
 
S. J. Pan and Q. Yang, “A survey on transfer learning,” IEEE
Transactions on knowledge and data engineering , vol. 22, no. 10, pp.
1345–1359, 2009.

 

 
 [354] 
 
S. J. Pan, V. W. Zheng, Q. Yang, and D. H. Hu, “Transfer learning for
WiFi-based indoor localization,” in Association for the advancement
of artificial intelligence (AAAI) workshop , vol. 6. The Association for the Advancement of Artificial
Intelligence Palo Alto, 2008.

 

 
 [355] 
 
S. J. Pan, J. T. Kwok, Q. Yang, and J. J. Pan, “Adaptive localization in a
dynamic WiFi environment through multi-view learning,” in AAAI ,
2007.

 

 
 [356] 
 
H. Zou, Y. Zhou, H. Jiang, B. Huang, L. Xie, and C. Spanos, “Adaptive
localization in dynamic indoor environments by transfer kernel learning,” in
 2017 IEEE wireless communications and networking conference
(WCNC) . IEEE, 2017, pp. 1–6.

 

 
 [357] 
 
F. Dou, J. Lu, Z. Wang, X. Xiao, J. Bi, and C.-H. Huang, “Top-down indoor
localization with Wi-Fi fingerprints using deep Q-Network,” in
 2018 IEEE 15th International Conference on Mobile Ad Hoc and Sensor
Systems (MASS) . IEEE, 2018, pp.
166–174.

 

 
 [358] 
 
J. Torres-Sospedra, R. Montoliu, A. Martínez-Usó, J. P. Avariento,
T. J. Arnau, M. Benedito-Bordonau, and J. Huerta, “UJIIndoorLoc: A new
multi-building and multi-floor database for WLAN fingerprint-based indoor
localization problems,” in 2014 international conference on indoor
positioning and indoor navigation (IPIN) . IEEE, 2014, pp. 261–270.

 

 
 [359] 
 
Intel Corporation,
 https://www.intel.com/content/www/us/en/products/docs/wireless-products/ultimate-n-wifi-link-5300-brief.html?wapkw=intel%20wifi%20link%205300 ,
accessed: December 2020.

 

 
 [360] 
 
Intel Corporation,
 https://ark.intel.com/content/www/us/en/ark/products/86068/intel-dual-band-wireless-ac-8260.html ,
accessed: December 2020.

 

 
 [361] 
 
DecaWave Limited,
 https://www.decawave.com/sites/default/files/resources/dw1000-datasheet-v2.09.pdf ,
accessed: December 2020.

 

 
 [362] 
 
L. Huang, X. Gan, B. Yu, H. Zhang, S. Li, J. Cheng, X. Liang, and B. Wang, “An
innovative fingerprint location algorithm for indoor positioning based on
array pseudolite,” Sensors , vol. 19, p. 4420, 10 2019.

 

 
 [363] 
 
W. I. Remcom,
“https://www.remcom.com/wireless-insite-em-propagation-softwareinsite-wireless-em-propagation-more-info,”
online, accessed: March 2017.

 

 
 [364] 
 
G. Xiong, T. Kim, and E. Perrins, “Decorrelation deep learning for
fingerprint-based indoor localization,” arXiv preprint
arXiv:1908.02014 , 2019.

 

 
 [365] 
 
X. Wang, L. Gao, and S. Mao, “BiLoc: Bi-modal deep learning for indoor
localization with commodity 5GHz WiFi,” IEEE access , vol. 5, pp.
4209–4220, 2017.

 

 
 [366] 
 
S. Dayekh, S. Affes, N. Kandil, and C. Nerguizian, “Cooperative localization
in mines using fingerprinting and neural networks,” in 2010 IEEE
Wireless Communication and Networking Conference . IEEE, 2010, pp. 1–6.

 

 
 [367] 
 
X. Wang, X. Wang, S. Mao, J. Zhang, S. C. Periaswamy, and J. Patton,
“DeepMap: Deep Gaussian Process for indoor radio map construction and
location estimation,” in 2018 IEEE Global Communications Conference
(GLOBECOM) . IEEE, 2018, pp. 1–7.

 

 
 [368] 
 
J. Liu, N. Liu, Z. Pan, and X. You, “AutLoc: deep autoencoder for indoor
localization with RSS fingerprinting,” in 2018 10th International
Conference on Wireless Communications and Signal Processing (WCSP) . IEEE, 2018, pp. 1–6.

 

 
 [369] 
 
A. Belay, L. Yen, H.-P. Lin, Y. Yayeh, Y. Li, S.-S. Jeng, and G. Berie,
“Applying deep neural network (DNN) for large-scale indoor localization
using feed-forward neural network (FFNN) algorithm,” 04 2018, pp.
814–817.

 

 
 [370] 
 
M. Nowicki and J. Wietrzykowski, “Low-effort place recognition with wifi
fingerprints using deep learning,” in International Conference
Automation . Springer, 2017, pp.
575–584.

 

 
 [371] 
 
W. I. Remcom, “http://yann.lecun.com/exdb/mnist/,” online, accessed: Oct
2020.

 

 
 [372] 
 
G. M. Mendoza-Silva, P. Richter, J. Torres-Sospedra, E. S. Lohan, and
J. Huerta, “Long-term wifi fingerprinting dataset for research on robust
indoor positioning,” Data , vol. 3, no. 1, p. 3, 2018.

 

 
 [373] 
 
G. M. Mendoza-Silva, P. Richter, J. Torres-Sospedra, E. S. Lohan, and
J. Huerta, “Long-Term Wi-Fi fingerprinting dataset and supporting
material,” Apr. 2020. [Online]. Available:
 https://doi.org/10.5281/zenodo.3748719 

 

 
 [374] 
 
E. S. Lohan, J. Torres-Sospedra, H. Leppäkoski, P. Richter, Z. Peng, and
J. Huerta, “Wi-fi crowdsourced fingerprinting dataset for indoor
positioning,” Data , vol. 2, no. 4, p. 32, 2017.

 

 
 [375] 
 
E. S. Lohan, J. Torres-Sospedra, P. Richter, H. Leppäkoski, J. Huerta, and
A. Cramariuc, “Crowdsourced WiFi database and benchmark software for indoor
positioning,” https://doi.org/10.5281/zenodo.889798 , Sep. 2017.
[Online]. Available: https://doi.org/10.5281/zenodo.889798 

 

 
 [376] 
 
A. Makki and C. J. Bleakley, “WLAN indoor ranging dataset for evaluation of
time of arrival estimation algorithms,” in The Ninth International
Conference on Indoor Positioning and Indoor Navigation (IPIN 2018), Nantes,
France, 24-27 September 2018 . IEEE,
2018.

 

 
 [377] 
 
M. A. and C. J. Bleakley, “WLAN Indoor Ranging Dataset,”
 https://www.kaggle.com/ahmedmakki/toa-dataset , May 2018. [Online].
Available: https://www.kaggle.com/ahmedmakki/toa-dataset 

 

 
 [378] 
 
M. Aernouts, R. Berkvens, K. Van Vlaenderen, and M. Weyn, “Sigfox and
LoRaWAN datasets for fingerprint localization in large urban and rural
areas,” Data , vol. 3, no. 2, p. 13, 2018.

 

 
 [379] 
 
M. Aernouts, R. Berkvens, K. Van Vlaenderen, and M. Weyn, “Sigfox and LoRaWAN
Datasets for Fingerprint Localization in Large Urban and Rural Areas,”
 https://doi.org/10.5281/zenodo.1193563 , Mar. 2018. [Online]. Available:
 https://doi.org/10.5281/zenodo.1193563 

 

 
 [380] 
 
D. Byrne, M. Kozlowski, R. Santos-Rodriguez, R. Piechocki, and I. Craddock,
“Data descriptor: Residential wearable RSSI and accelerometer measurements
with detailed location annotations.”

 

 
 [381] 
 
D. Byrne, M. Kozlowski, R. Santos-Rodriguez, R. Piechocki, and I. Craddock,
“Residential wearable rssi and accelerometer measurements with detailed
location annotations,” Scientific data , vol. 5, p. 180168, 2018.

 

 
 [382] 
 
J. Torres-Sospedra, D. Rambla, R. Montoliu, O. Belmonte, and J. Huerta,
“Ujiindoorloc-mag: A new database for magnetic field-based localization
problems,” in 2015 International conference on indoor positioning and
indoor navigation (IPIN) . IEEE, 2015,
pp. 1–10.

 

 
 [383] 
 
D. Dua and C. Graff, “UCI machine learning repository,” 2017. [Online].
Available: http://archive.ics.uci.edu/ml 

 

 
 [384] 
 
J. Torres-Sospedra, D. Rambla, R. Montoliu, O. Belmonte, and J. Huerta,
“UJIIndoorLoc-Mag: A New Database for Magnetic Field-Based Localization
Problems,” https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc-Mag ,
2015. [Online]. Available:
 https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc-Mag 

 

 
 [385] 
 
J. Torres-Sospedra, R. Montoliu, A. Martínez-Usó, J. P. Avariento,
T. J. Arnau, M. Benedito-Bordonau, and J. Huerta, “Ujiindoorloc: A new
multi-building and multi-floor database for wlan fingerprint-based indoor
localization problems,” in 2014 international conference on indoor
positioning and indoor navigation (IPIN) . IEEE, 2014, pp. 261–270.

 

 
 [386] 
 
J. Torres-Sospedra, R. Montoliu, A. Martínez-Usó, J. P. Avariento,
T. J. Arnau, M. Benedito-Bordonau, and J. Huerta, “UJIIndoorLoc: A New
Multi-building and Multi-floor Database for WLAN Fingerprint-based Indoor
Localization Problems,”
 http://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc , 2014. [Online].
Available: http://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc 

 

 
 [387] 
 
A. Popleteev, “AmbiLoc: A year-long dataset of FM, TV and GSM fingerprints
for ambient indoor localization,” in 8th International Conference on
Indoor Positioning and Indoor Navigation (IPIN-2017) , 2017.

 

 
 [388] 
 
A. Popleteev, “AmbiLoc: A year-long dataset for ambient indoor
localization,” https://ambiloc.org/ , Sept 2017. [Online]. Available:
 https://ambiloc.org/ 

 

 
 [389] 
 
N. Moayeri, M. O. Ergin, F. Lemic, V. Handziski, and A. Wolisz, “PerfLoc
(part 1): An extensive data repository for development of smartphone indoor
localization apps,” in 2016 IEEE 27th Annual International Symposium
on Personal, Indoor, and Mobile Radio Communications (PIMRC) . IEEE, 2016, pp. 1–7.

 

 
 [390] 
 
C. Laoudias, R. Piché, and C. G. Panayiotou, “Device self-calibration in
location systems using signal strength histograms,” Journal of
Location Based Services , vol. 7, no. 3, pp. 165–181, 2013.

 

 
 [391] 
 
C. Laoudias, R. Piché, and C. Panayiotou, “KIOS WiFi RSS dataset,”
 http://goo.gl/u7IoG , 09 2013. [Online]. Available:
 http://goo.gl/u7IoG 

 

 
 [392] 
 
G. M. Mendoza-Silva, M. Matey-Sanz, J. Torres-Sospedra, and J. Huerta, “BLE
RSS measurements dataset for research on accurate indoor positioning,”
 Data , vol. 4, no. 1, p. 12, 2019.

 

 
 [393] 
 
G. M. Mendoza-Silva, M. Matey-Sanz, J. Torres-Sospedra, and J. Huerta, “BLE
RSS measurements database and supporting materials,”
 https://doi.org/10.5281/zenodo.1618692 , Dec. 2018. [Online]. Available:
 https://doi.org/10.5281/zenodo.1618692 

 

 
 [394] 
 
P. Roy, C. Chowdhury, D. Ghosh, and S. Bandyopadhyay, “JUIndoorLoc: A
ubiquitous framework for smartphone-based indoor localization subject to
context and device heterogeneity,” Wireless Personal Communications ,
vol. 106, no. 2, pp. 739–762, 2019.

 

 
 [395] 
 
Q. Tian, I. Kevin, K. Wang, and Z. Salcic, “Human body shadowing effect on
UWB-based ranging system for pedestrian tracking,” IEEE Transactions
on Instrumentation and Measurement , vol. 68, no. 10, pp. 4028–4037, 2018.

 

 
 [396] 
 
X. Chen, L. Tian, P. Tang, and J. Zhang, “Modelling of human body shadowing
based on 28 GHz indoor measurement results,” in 2016 IEEE 84th
vehicular technology conference (VTC-Fall) . IEEE, 2016, pp. 1–5.

 

 
 [397] 
 
C. Cziezerski, M. Heino, P. Koivumaki, K. Haneda, C. Icheln, A. Hazmi, and
R. Tian, “Comparing gains of 28 GHz module-based phased antenna arrays on
a 5G mobile phone,” 2019.

 

 
 [398] 
 
J. Karedal, A. J. Johansson, F. Tufvesson, and A. F. Molisch, “Shadowing
effects in MIMO channels for personal area networks,” in IEEE
Vehicular Technology Conference . IEEE, 2006, pp. 1–5.

 

 
 [399] 
 
K. C. Chen, S. C. Lin, J. H. Hsiao, C. H. Liu, A. F. Molisch, and
G. P. Fettweis, “Wireless networked multirobot systems in smart
factories,” Proceedings of the IEEE , pp. 1–27, 2020.

 

 
 [400] 
 
Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, and S. Y. Philip, “A comprehensive
survey on graph neural networks,” IEEE Transactions on Neural Networks
and Learning Systems , 2020.