A Review of Speech-centric Trustworthy Machine Learning: Privacy, Safety, and Fairness 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2212.09006v2 [cs.SD] 16 Apr 2023 
 
 

# A Review of Speech-centric Trustworthy Machine Learning: Privacy, Safety, and Fairness

 
 
 Tiantian Feng
 
    
 Rajat Hebbar † 
 
    
 Nicholas Mehlman † 
 
    
 Xuan Shi † 
 
    
 Aditya Kommineni † 
 
    
 Shrikanth Narayanan
 
 Affiliation: [
 

 Abstract 
 
 Speech-centric machine learning systems have revolutionized a number of leading industries ranging from transportation and healthcare to education and defense, fundamentally reshaping how people live, work, and interact with each other. However, recent studies have demonstrated that many speech-centric ML systems may need to be considered more trustworthy for broader deployment. Specifically, concerns over privacy breaches, discriminating performance, and vulnerability to adversarial attacks have all been discovered in ML research fields. In order to address the above challenges and risks, a significant number of efforts have been made to ensure these ML systems are trustworthy, especially private, safe, and fair. In this paper, we conduct the first comprehensive survey on speech-centric trustworthy ML topics related to privacy, safety, and fairness. In addition to serving as a summary report for the research community, we highlight several promising future research directions to inspire researchers who wish to explore further in this area.

 
 
 
 1]University of Southern California, Los Angeles, USA; 
 email: tiantiaf@usc.edu
 \articledatabox † : Contribute equally in terms of text.

 
 \makeabstracttitle 
 
 

## Chapter 1 Introduction

 
 In the last few years, machine learning (ML), particularly deep learning, has empowered tremendous breakthroughs in a variety of research fields and applications, including natural language processing ( [ 51 ] ), image classification ( [ 87 ] ), video recommendation ( [ 49 ] ), healthcare analysis ( [ 125 ] ), and even mastering the chess game ( [ 161 ] ). The deep learning model typically consists of multiple processing layers with a combination of both linear and non-linear computations. Although training a deep learning model with the multi-layer architecture demands the accumulation of massive datasets and access to large-scale computational infrastructures ( [ 20 ] ), the trained model usually achieves state-of-the-art (SOTA) performance compared to the traditional modeling approaches. The broad success of deep learning has enabled a more profound understanding of the human condition (state, trait, behavior, interaction) and revolutionized technologies that support and enhance human experiences. Alongside the success that ML has had in these areas, significant progress has also been made in speech-centric ML.

 
 
 Speech is a natural and prominent form of communication between humans that exists in almost every spectrum of human life, whether chatting with friends, discussing with colleagues, or having a remote call with the family. The advancement in speech-centric machine learning has enabled the ubiquitous usage of smart assistants such as Siri, Google Voice, and Alexa. In addition, speech-centric modeling has created numerous research topics in human behavior understanding, human-computer interface (HCI) ( [ 44 ] ), and social media analysis, involving several widely used speech modeling techniques like automatic speech recognition ( [ 115 ] ), speech emotion recognition ( [ 5 ] ), automatic speaker verification ( [ 93 ] ), and keyword spotting ( [ 191 ] ).

 
 
 Despite the prospect of the broad deployment of ML systems in a wide range of speech-centric applications, two intertwined challenges remain unaddressed in most of these systems: understanding and illuminating the rich diversity across people and contexts while creating trustworthy ML technologies that work for everyone in all contexts. Trust is fundamental in human life, whether to trust friends, colleagues, family members, or AI-powered services. While ML practitioners, such as researchers, and decision-makers, conventionally focus on improving the system performance of ML models using performance metrics such as the F1 score, ensuring ML application that is trustworthy stays a challenging topic. In the past few years, we have witnessed a significant amount of research work targeting trustworthy AI and ML, and the objective of this paper is to provide a comprehensive review of related research activities, with an emphasis on speech-centric ML. This survey aims to outline salient design pillars and the latest research trends in speech-centric trustworthy ML.

 
 
 Trustworthiness in ML has been defined differently across the literature. For example, [ 90 ] described the term trustworthiness based on the industry practices, involving the implementation of the certification process and explanation process. The certification process consists of testing and verification modules to detect potential fabrications or perturbations in the input data. The explanation refers to the ability to explain why ML reached a specific decision based on the input data. Furthermore, the ethics guidelines for trustworthy artificial intelligence published by EU ( [ 163 ] ) recognized an AI system, to be considered trustworthy, must comply with laws and regulations, adhere to ethical principles, and function robustly. More recently, [ 106 ] summarized the trustworthy AI from safety, fairness, explainability, privacy, accountability, and environmentally friendly aspects. Likewise, our review recognizes robustness, reliability, safety, security, inclusiveness, and equity as core design elements of the trustworthiness ML system. Based on these criteria, our paper surveys the literature on speech-centric trustworthy ML from privacy, safety, and fairness perspectives, as illustrated in Figure 1.1 1 1 
 1 
 
 
 
 The figure in this paper uses images from https://openmoji.org/ :

 
 
 Figure 1.1 : Summary of key factors contribute to speech-centric trustworthy machine learning: privacy, safety, and fairness. 
 
 
 Privacy: The speech-centric ML systems rely heavily on collecting speech data that is from, about, and for people in potentially sensitive environments and contexts, like homes, workplaces, hospitals, and schools. The collection of speech data frequently raises significant concerns about compromising user privacy, such as revealing sensitive information that people might want to keep private ( [ 104 ] ). It is critical to make sure that the speech data that is either shared by an individual or collected by an ML system is protected from unjustifiable and unauthorized uses.

 
 
 Safety: Over the past few years, it has been discovered that ML systems are susceptible to adversarial attacks that aim to exploit vulnerabilities in the model’s prediction function for malicious purposes ( [ 75 ] ). For example, by introducing minor perturbations to the speech data, a malicious actor can cause the keyword spotting model to drastically misclassify the desired input speech commands. Therefore, a trustworthy ML system must generate reliable outputs for the same inputs even if the inputs have been intentionally altered by the malicious attacker [ 119 ] .

 
 
 Fairness: Recently, it has come to light that ML systems can perform unfairly. Why an ML system mistreats people is multifold ( [ 120 ] ). One factor is the societal aspect, where the ML system generates biased outputs because of the societal biases in the training data or the assumptions/decisions throughout the ML development processes. Another reason that causes unfairness in AI is the imbalance of dataset characteristics, where limited data samples exist for some groups of people. As a result, the model needs to accommodate the needs of these groups to avoid biased outputs. It is also crucial to note that deploying unfair ML systems can amplify societal biases and data imbalance issues. To evaluate the trustworthiness of the speech-centric ML system, ML practitioners need to evaluate whether the ML model produces discriminatory outcomes towards individuals or groups.

 
 
 The remainder of this article is organized as follows. Section 2 briefly summarizes popular speech-centric tasks, datasets, and SOTA modeling frameworks. Section 3 overviews the related survey papers in trustworthy speech-centric applications. Section 4 comprehensively discusses safety considerations in the speech-centric ML system. Section 5 discusses privacy risks and defenses in speech modeling. Section 6 reviews the emerging fairness issues with speech modeling tasks. Section 7 elaborates on potential developments and challenges in the future for speech-centric trustworthy machine learning. Finally, section 8 concludes with a summary of the key observations in this article. Specifically, our contributions are summarized as follows:

 
 
 
 1. 
 
 To the best of our knowledge, this is the first review work to provide a comprehensive review of designing trustworthy ML focused on speech-centric modeling. We survey most, if not all, the published and pre-print works covering automatic speech recognition, speech emotion recognition, keyword spotting, and automatic speaker verification.

 

 2. 
 
 We create the taxonomies to systematically review design pillars related to the trustworthiness of speech-centric ML systems. We further compare a variety of literature on each key factor.

 

 3. 
 
 We discuss the outstanding challenges of designing speech-centric ML systems facing trustworthiness considerations related to privacy, safety, and fairness. Based on the literature reviewed, we also discuss the challenges yet to be solved and suggest several promising future directions.

 

 
 
 
 

## Chapter 2 Speech-centric Machine Learning

 
 To begin with, we introduce the fundamentals of speech-centric ML systems to offer the audience a more in-depth understanding of the trustworthy aspect of such systems. Our aim is to provide a concise summary of speech-centric ML tasks, commonly used speech datasets, and speech modeling approaches.

 
 

### 2.1 Speech-centric ML tasks

 
 Automatic Speech Recognition (ASR) : ASR system is one of the most prominent tasks in the speech domain, used to convert human speech into readable text. ASR techniques have found wide applications in modern human-computer interactive scenarios, such as Siri and Alexa. An ASR model typically involves pre-processing, feature extraction, classification, and the language model. The most widely adopted speech processing methods include framing, normalization, and pre-emphasis. After pre-processing, a feature extraction module extracts speech features as the input to the classifier. Mel-frequency cepstral coefficients (MFCCs) and Mel Spectrogram are commonly used speech features. The classification then predicts the spoken text based on the input features. Finally, a language model may be used to recognize the phoneme predicted by the classification model. It is worth noting that language models can significantly increase the efficiency of ASR systems, but it is not a necessary component in ASR systems. Many modern ASR systems can still function without using language models. Notably, word error rate (WER) is a standard metric to measure the performance of an ASR system, calculated as the number of errors in transcripts divided by the total number of spoken words. Readers interested in ASR systems may refer to [ 115 ] for a systematic introduction.

 
 
 Speech Emotion Recognition (SER) : The task of speech emotion recognition involves the classification of emotions (such as neutral, happy, sad, and angry) from speech signals, where the emotion labels are typically obtained from multiple human annotators. Equivalent to ASR, SER systems extract speech features passing through a classifier for emotion prediction. Due to the imbalanced label distributions in many existing SER datasets, the unweighted average recall (UAR) is often used as the evaluation metric for SER systems. For readers interested in a more comprehensive introduction to SER systems, we recommend referring to [ 5 ] .

 
 
 Automatic speaker verification (ASV) : ASV aims to determine the authorization of the claimed identity based on voice fingerprint. ASV is composed of two stages: the enrolment and the testing stage. During the enrolment stage, embeddings or latent representations are extracted from speakers and stored. In testing, embeddings from a test speaker are extracted and compared to the claimed speaker’s embeddings. Verification pairs from the same speaker are referred to as genuine pairs, while pairs from different speakers are "imposter" pairs ( [ 139 ] ). Equal error rate (EER) is one of the most common metrics to evaluate the performance of ASV systems, which is the value at which the false acceptance rate (FAR) equals the false rejection rate (FRR). Further information on recent ASV developments can be found in [ 93 ] .

 
 
 Keyword Spotting (KWS) : Keyword spotting refers to the detection of predefined keywords or phrases in a speech recording, such as "Hello, Siri." This speech task is widely adopted in commercially available smart assistants such as Apple’s Siri, Google Assistant, and Amazon’s Alexa. Compared to ASR systems, KWS systems require substantially fewer computational resources and are well-suited for on-device learning.

 
 
 Table 2.1 : Table showing the List of speech datasets that can be used for training the ASR model, the SER model, the AVS model, and the KWS model. 
 
 
 
 Speech Task | 
 Dataset Name | 
 Year | 
 Hours | 

 
 | 
 Librispeech ( [ 136 ] ) | 
 2015 | 
 1,000 | 

 
 | 
 WSJ ( [ 70 ] ) | 
 1994 | 
 162 | 

 
 | 
 Common Voice ( [ 13 ] ) | 
 2019 | 
 1,900 | 

 
 ASR | 
 The CHiME-5 ( [ 17 ] ) | 
 2018 | 
 50 | 

 
 | 
 TED-LIUM ( [ 151 ] ) | 
 2012 | 
 452 | 

 
 | 
 The Spoken Wikipedia ( [ 99 ] ) | 
 2016 | 
 1,005 | 

 
 | 
 Voxpopuli ( [ 190 ] ) | 
 2021 | 
 1,800 | 

 
 | 
 IEMOCAP ( [ 28 ] ) | 
 2008 | 
 12 | 

 
 | 
 CMU-MOSEI ( [ 201 ] ) | 
 2018 | 
 65 | 

 
 SER | 
 MSP-Podcast ( [ 117 ] ) | 
 2020 | 
 100 | 

 
 | 
 MELD ( [ 141 ] ) | 
 2018 | 
 - | 

 
 | 
 ESD ( [ 206 ] ) | 
 2021 | 
 29 | 

 
 | 
 Voxceleb1 ( [ 130 ] ) | 
 2017 | 
 352 | 

 
 AVS | 
 Voxceleb2 ( [ 42 ] ) | 
 2018 | 
 2,442 | 

 
 | 
 VoxMovies ( [ 27 ] ) | 
 2021 | 
 - | 

 
 KWS | 
 Speech Commands ( [ 191 ] ) | 
 2018 | 
 - | 

 

 
 
 

### 2.2 Speech Datasets

 
 The speech datasets are fundamental for training and testing speech-centric ML systems. This section provides a brief overview of the commonly used datasets for various downstream speech tasks. These datasets are also frequently used as benchmarks for evaluating speech-centric trustworthy ML. Then, we summarize the datasets along with their published years and total recording time in Table 2.1 .

 
 
 ASR : Among all the datasets listed in Table 2.1 , Librispeech ( [ 136 ] ) is one of the most commonly used datasets for training and evaluating ASR algorithms. This dataset includes 1000 hours of audiobooks and transcriptions. Common Voice ( [ 13 ] ) is another widely used dataset supported by Mozilla. The dataset is entirely open source, and any people can contribute to the dataset by providing their speech recordings. Moreover, the CHiME-5 ( [ 17 ] ) and TED-LIUM ( [ 151 ] ) datasets consist of audio recordings from the home environment and TED talks.

 
 
 SER : One of the most frequently referred datasets in the SER field is the IEMOCAP dataset ( [ 28 ] ). This dataset consists of audio-visual data collected from 10 actors engaged in dyadic interactions. Each recorded utterance is provided with annotations into categorical emotion labels and transcripts. In addition to IEMOCAP, other commonly used datasets include the CMU-MOSEI dataset, which is based on YouTube videos ( [ 201 ] ), the MSP-Podcast dataset, which is based on podcasts ( [ 117 ] ), and the MELD dataset, which is based on movies ( [ 141 ] ).

 
 
 AVS : The two most frequently used datasets for training the AVS system are Voxceleb1 ( [ 130 ] ) and Voxceleb2 ( [ 42 ] ). The voxceleb1 dataset contains speech data from over 1000 celebrities on Youtube. Shortly after the success of the voxceleb1 dataset, the authors introduced the voxceleb2 dataset which includes over 6000 celebrities’ speech data on Youtube.

 
 
 KWS : Surprisingly, there are few purposely designed datasets for this downstream task. The Google Speech Commands dataset ( [ 191 ] ) is the most commonly used KWS dataset that includes 35 frequently used spoken words from the everyday vocabulary. The dataset includes a total of 105,829 audio recordings from 2,618 speakers, wherein 2,112 speakers are in the training set and the rest are in the test set.

 
 
 

### 2.3 Speech Modeling Approach

 

#### 2.3.1 Conventional Modeling Approach

 
 Conventional speech modeling systems heavily relied on the Gaussian mixture model (GMM), Hidden Markov Model (HMM), and the support vector machine (SVM). For example, traditional speaker verification systems have used the Gaussian mixture model based universal background model (GMM-UBM) since early 2000 ( [ 148 ] ). Later, the researchers proposed the i-vector framework ( [ 50 ] ), which reduced the high-dimensional GMM-UBM supervectors into low-dimensional vectors using factor analysis. The i-vector has been a SOTA technique in ASV systems for many years. In addition, HMM was one of the most widely used ASR modeling approaches before the deep learning approach gained popularity ( [ 181 ] ). On the other hand, statistical descriptors of low-level speech features (e.g., pitch, intensity) were predominantly applied in emotion recognition and sentiment analysis tasks. OpenSMILE ( [ 61 ] ) is one of the widely adopted tools that enable abundant research works using this approach. However, the performance of these traditional modeling frameworks is largely impacted by the channel variations and utterance variations of the input speech signals.

 
 
 

#### 2.3.2 Deep Learning Approach

 
 In recent years, we have seen a wide variety of successes in applying deep learning to speech-centric systems. Deep speech ( [ 83 ] ) is one of the most recognized deep neural models based on the recurrent neural network (RNN). This model has been a strong baseline in the ASR domain for years. Until a few years ago, with the popularity of transformer architecture ( [ 185 ] ), researchers have proposed the self-supervised learning framework like Wav2Vec 2.0 ( [ 16 ] ). Wav2Vec 2.0 has quickly become the SOTA machine learning model for ASR ( [ 16 ] ) and SER ( [ 36 ] ). This model even achieves similar performance in the ASR task compared to humans. Around the same time, Google proposed a convolution-augmented transformer called Conformer ( [ 79 ] ) that reached competitive ASR performance to the Wav2Vec 2.0 model. More recently, OpenAI introduced its transformer-based ASR model called Whisper ( [ 145 ] ), which outperformed most existing works. Similar to the ASR task, many SOTA speaker verification systems are built upon the deep embedding, also known as x-vector, extracted from the deep neural network ( [ 164 ] ).

 
 
 
 
 

## Chapter 3 Related Surveys in Trustworthy Speech-centric Machine Learning

 
 In this section, we describe several related survey papers that cover the privacy and adversarial defense in speech applications. However, to the best of our knowledge, there are no survey papers focusing on fairness risks in speech-centric applications.

 
 

### 3.1 Privacy

 
 [ 29 ] provides a detailed literature review on generative models, while the author briefly describes the use of generative speech models to protect user privacy. [ 29 ] introduces 2 usage cases, remote health monitoring, and voice assistance, based on Generative speech models. In both applications, the author presents GAN-based speech models that either obfuscate the sensitive attribute or transform the user voice into a common speech signal. [ 39 ] is another review paper that focuses on privacy in personal voice assistant applications. This survey paper discusses voice privacy preservation mechanisms including encryption schemes, voice anonymization, and distributed learning. However, this paper does not provide comprehensive reviews on privacy attacks and also does not summarize the taxonomy of privacy-preserving methods.

 
 
 

### 3.2 Safety

 
 Apart from surveying privacy-related research topics, [ 39 ] provides extensive reviews on security challenges in speech-centric applications. The author discusses a wide range of research works in mitigating adversarial attacks in ASR applications. This review covers popular adversarial attacks including PGD attacks, gradient decent attacks, etc. However, this review does not provide a categorization of adversarial attacks. Likewise, [ 92 ] provides a survey on the safety aspect in speech-centric applications that categorize adversarial attacks based on types of attacking perturbations and threat models. Despite surveying different adversarial attacks in speech-centric applications, these survey papers do not discuss theories behind adversarial attacks and mitigations.

 
 
 

### 3.3 Fairness

 
 Although many survey papers have discussed the fairness considerations in machine learning (e.g, [ 120 ] ), there is a lack of comprehensive literature studying bias and fairness in speech-centric applications. Most previous work in the speech domain focus on specific applications such as ASR, ASV and SER using traditional evaluation schemes.
For example, [ 139 ] highlight the need for fairness-centric metrics that better highlights the bias in existing methods and evaluation schemes to improve fairness in ASV.
In this paper, we provide detailed reviews summarizing the recent works that target improving model fairness in ASR, ASV, and SER applications.

 
 
 
 

## Chapter 4 Safety in Speech-centric Machine Learning

 
 Safety concerns in ML systems originated from adversarial attacks. Adversarial attacks involve a malicious actor (‘adversary’) who manipulates data samples with the intention of negatively affecting model performance. In a poisoning attack, the adversary manipulates data and/or model parameters during training, while in an evasion attack, they pass an adversarial sample to the model at evaluation time. In both scenarios, the adversary attempts to avoid detection by the model’s users and/or administrators.

 
 

### 4.1 Evasion Attacks and Defenses

 

#### 4.1.1 Preliminaries

 
 Evasion attacks can be classified both by the adversary’s knowledge of the targeted model (black box or white box) and by the attack’s objective (targeted or untargeted). A black-box attacker has no special knowledge of the model other than being able to observe its predictions. On the other hand, a white-box attacker possesses full visibility into model architecture, parameters, ex., and importantly, can perform backpropagation to extract loss function gradients. The white-box scenario is generally viewed as the more insidious threat model. An untargeted attack tries to produce incorrect predictions on a malicious sample without regard to the exact nature of the miss-classification. Mathematically, given benign input x x with true label y y , the untargeted attack optimizes for a perturbation δ \delta that satisfies:

 
 
 

 
 | 
 δ = argmax ​ ℒ ​ ( x + δ , y ) ​ s . t . ‖ δ ‖ ϵ \delta=\mathrm{argmax}\ \mathcal{L}(x+\delta,y)\ \mathrm{s.t.}\ \|\delta\| \epsilon | 
 | 
 (4.1) | 
 

 
 
 where ℒ \mathcal{L} is a loss function, and ∥ ⋅ ∥ \|\cdot\| is some norm. The norm constraint is needed to minimize the chance of detection. In contrast, a targeted attack seeks miss-classification as a specific incorrect class. For benign input x x with associated label y y , the adversary solves

 
 
 

 
 | 
 δ = argmin ​ ℒ ​ ( x + δ , y ^ ) ​ s . t . ‖ δ ‖ ϵ \delta=\mathrm{argmin}\ \mathcal{L}(x+\delta,\hat{y})\ \mathrm{s.t.}\ \|\delta\| \epsilon | 
 | 
 (4.2) | 
 

 
 
 where y ^ \hat{y} is the targeted label for the (incorrect) class the adversary wants the model to predict. Once again the constraint ‖ δ ‖ ϵ \|\delta\| \epsilon ensures that the attack is relatively imperceptible to the human user.

 
 
 Some of the earliest work on evasion attacks was presented by [ 22 ] . They formulated the adversary’s objective in terms of a constrained optimization problem in which the attacker searches for a perturbation that successfully ‘fools’ the target model, while simultaneously ensuring that the attacked example remained within an ϵ \epsilon -bound of the original. A gradient descent approach was proposed to generate these adversarial examples. A similar framework was suggested by [ 173 ] , although L-BFGS was used in place of gradient descent. Additionally, [ 173 ] demonstrated the transferability of adversarial examples across deep learning models (i.e. attacks generated for one model can successfully fool a different one). [ 75 ] link attack transferability to excessive model linearity, hypothesizing that successful perturbations are highly aligned with weight vectors, and thus tend to generalize well. They also introduce the fast gradient sign method ‘FGSM’ for quickly constructing untargeted adversarial examples. For an FGSM attack with an ℓ ∞ \ell_{\infty} norm constraint of size ϵ \epsilon , the adversarial perturbation δ \delta for sample x x is given by δ = ϵ ​ sign ​ ( ∇ x J ​ ( x , y , θ ) ) \delta=\epsilon\ \mathrm{sign}(\nabla_{x}J(x,y;\theta)) . Here J J is some loss function, y y is the true label for x x , and θ \theta represents the model’s parameters. [ 114 ] extended the FGSM methodology to introduce the Project Gradient Descent (PGD) attack. PGD employs an iterative approach to craft adversarial examples, with each step consisting of an FGSM perturbation followed by projection onto the ϵ \epsilon -ball. PGD attacks can be used in both targeted and untargeted threat models.

 
 
 

#### 4.1.2 Evasion Attacks in Speech-centric ML

 
 A number of speech-specific attacks have also been proposed. [ 74 ] provided one of the first investigations of end-to-end gradient-based white box attacks on audio-modality classifiers (e.g. speaker recognition). They demonstrated that the attacks substantially degrade model accuracy while only minimally impacting human perceptual evaluation. [ 35 ] also presented an attack against speaker recognition models but used a black-box approach that relies on gradient estimation. Black-box approaches to attacking ASR systems have been introduced by [ 1 ] , [ 10 ] , and [ 98 ] .

 
 
 Meanwhile, [ 31 ] presented a strong targeted white box attack against ASR, that obtains a perfect success rate (w.r.t. the ability of the attack to produce the targeted transcript) with greater than 30 30 dB mean adversarial SNR. Their approach specified the generic adversarial framework for the Connectionist Temporal Classification (CTC) loss commonly used for training ASR systems. They also quantify perturbation magnitude in decibels instead of linear units. Within the framework from [ 31 ] , attack generation equates to solving the following optimization problem

 
 
 

 
 | 
 min ⁡ | δ | 2 2 + α ​ ℒ CTC ​ ( x + δ , t ) ​ s . t . dB ⁡ ( δ ) − dB ⁡ ( x ) τ \min|\delta|_{2}^{2}+\alpha\mathcal{L}_{\mathrm{CTC}}(x+\delta,t)\ \mathrm{s.t.}\ \mathrm{dB}(\delta)-\mathrm{dB}(x) \tau | 
 | 
 

 
 
 where t t is the targeted transcription and ℒ CTC \mathcal{L}_{\mathrm{CTC}} is the CTC loss function. By optimizing this objective with successively smaller values of τ \tau , the adversary determines the smallest magnitude perturbation that achieves the targeted objective. This attack methodology was extended by [ 144 ] which introduced the ‘Imperceptible attack’ for ASR. After finding a perturbation that fools the network, an additional update step was used to regularize the power spectrum of the attack such that it falls under the masking threshold of the clean speech. These two steps (perturbation update and spectral shaping) were repeated for a number of iterations. In this manner, the Imperceptible attacks leveraged audio-specific notions of perceptibility in constraining the attack generation process. Similar work on psychoacoustically motivated attacks has also been presented in [ 155 ] .

 
 
 

#### 4.1.3 Defenses against Evasion Attacks in Speech-centric ML

 
 A variety of defenses have been proposed to counteract the insidious effects of evasion attacks. One method that has been shown to be broadly successful is adversarial training (AT), in which adversarial examples are generated during the training process and used to further tune the model weights ( [ 75 , 114 ] ) 1 1 
 1 
 
 
 
 Note that [ 114 ] investigates both attacks (PGD) and defenses (AT) . This approach is not unlike traditional data augmentation methods such as additive noise or image cropping, except that the adversarial samples are generated specifically for the model at hand (usually using a white-box attack such as FGSM). Unfortunately, despite its success, AT introduces a significant computational overhead since it requires the construction of adversarial examples on each training epoch, as well as additional weight updates.

 
 
 Another widely known defense is Randomized Smoothing (RS), first introduced by [ 45 ] . RS uses stochastic averaging to ‘wash-out’ the effects of adversarial perturbations, which are relatively small in absolute magnitude. Specifically, given a classifier f ⁡ ( ⋅ ) f(\cdot) that is robust to additive Gaussian noise, a new ‘smoothed’ classifier g ⁡ ( ⋅ ) g(\cdot) is produced by g ⁡ ( x ) = argmax k ​ P ​ ( f ⁡ ( x + ϵ ) = k ) g(x)=\mathrm{argmax}_{k}\ P(f(x+\epsilon)=k) with ϵ ∼ 𝒩 ⁡ ( 0 , σ 2 ​ I ) \epsilon\sim\mathcal{N}(0,\sigma^{2}I) . Random smoothing can provide provable robustness guarantees to ℓ 2 \ell_{2} bounded attacks (see [ 45 ] ).

 
 
 Other defenses include changes to model architecture and training procedures. For example, a modified RELU activation function that constrains the maximum value of a given neuron was suggested by [ 202 ] . Similarly, [ 43 ] introduced Parseval Regularization which was based on minimizing the Lipschitz constant. Other works, such as [ 103 ] and [ 134 ] , have proposed the use of denoisers of other preprocessor methods to remove or mitigate the impact of adversarial perturbations.

 
 
 Speech-specific defenses include the detection framework for ASR attacks from [ 91 ] which leverages the sensitivity of adversarial examples to small changes. They identified adversarial samples by testing whether the predicted transcript varied substantially under alterations such as filtering and quantization. [ 198 ] used a similar approach for attack detection that tests the consistency between the transcript prediction for the full audio signal and the prediction for a partial segment. Several papers have also proposed audio resyntheses defenses, such as the GAN-based methods from [ 59 ] and [ 60 ] .

 
 
 Table 4.1 : Summary of key works on adversarial attacks 
 
 
 
 | 
 Evasion Attacks | 
 Poisoning Attacks | 

 
 
 
 | 
 [ 75 ] | 
 [ 23 ] | 

 
 | 
 [ 114 ] | 
 [ 129 ] | 

 
 Attacks | 
 [ 31 ] | 
 [ 108 ] | 

 
 | 
 | 
 [ 78 ] | 

 
 | 
 [ 75 ] | 
 [ 169 ] | 

 
 Defenses | 
 [ 45 ] | 
 [ 179 ] | 

 
 | 
 [ 43 ] | 
 [ 69 ] | 

 

 
 
 
 

### 4.2 Poisoning Attacks and Defenses

 

#### 4.2.1 Preliminaries

 
 Whereas an evasion attack occurs at inference time, poisoning attacks involve corruption of the training data thus leading to an inaccurate or vulnerable model. Given the increasing use of publicly sourced data and federated learning, it is often impossible to guarantee the integrity of a given dataset. The authors of [ 18 ] provided one of the earliest investigations of poisoning attacks. In particular, they distinguish between availability attacks that attempt to a broadly inaccurate model and integrity attacks that seek to produce a more specific vulnerability (e.g., misclassification in response to the presence of a specific trigger). In both scenarios, the attacker has the ability to manipulate the features and/or labels for some small fraction of the training data to decrease the performance of the learned model.

 
 
 A variety of approaches have been proposed to generate availability-type poisoning attacks. In this case, the adversarial objective is to generate a set of poisoned samples that minimizes the learned model’s performance on a withheld set of test samples. [ 23 ] presented an attack on the SVM classifier that used gradient methods to generate the poisoned samples. SVM poisoning attacks have also been studied by [ 121 ] in addition to attacks against linear and logistic regression. The Karush-Kuhn-Tucker (KKT) optimal conditions to generate the poisoned samples. While the relative simplicity of these models makes direct optimization feasible, Deep Neural Networks require different approaches. For example, the authors of [ 129 ] introduced a method that estimates back-propagation through the entire training procedure to efficiently generated poisoned samples to maximize the validation loss in the final model. Another approach suggested by [ 196 ] was to generative poisoned samples with the Generative Adversarial Network (GAN) where the target model acted as the discriminator and a separate auto-encoder model was used as the generator that synthesized the poisoned samples.

 
 
 A large number of integrity-type poisoning attacks attempt to produce a model that performs normally on clean test samples while miss-classifying those that contain a specific trigger. For example, [ 108 ] showed how a pre-trained facial recognition model can be ‘Trojaned’ by creating a trigger-induced backdoor. They first used the model gradient to generate a trigger patch that produced a large-magnitude activation for some hidden layer neurons. Then they reconstructed faux training examples for each class by performing gradient ascent on the input image. These synthetic samples form the basis for a finetuning procedure, in which a subset of the samples have the trigger added and the label changed. The resulting model should then miss-classify future samples containing the trigger. Importantly, this is accomplished without direct knowledge of the training dataset. A similar threat model was introduced by [ 78 ] , which demonstrated the efficacy of the approach by creating a poisoned model that incorrectly identified street signs when a small trigger was present.

 
 
 While both [ 108 ] and [ 78 ] assumed some attack autonomy over model training, other work such as by [ 38 ] , assumed that the attacker only has access to a small percentage of the training data. The authors proposed several approaches to generate backdoor samples, including an input-instance-key method in which the poisoned samples are randomly perturbed versions of a ‘key’ input with the target label. The adversary hoped that the final model would thus misclassify test samples that are similar to the key input used to poison the training data. Another poisoning algorithm suggested by [ 108 ] was to create poisoned instances by adding a specific pattern to benign samples, and changing the label to the targeted class. A clean-label attack that manipulated only the input features was presented by [ 156 ] . Their approach optimized for an adversarial sample that is perceptually similar to the target class in input space but close to a target instance in feature space. If the target instance is then deployed on the trained model, it is more likely to be misclassified as the target class. Another clean label attack was the Witches Brew method by [ 72 ] , which used gradient matching to produce samples that modified the model’s training trajectory such that it would misclassify a specific target instance at test time.

 
 
 

#### 4.2.2 Poisoning Attacks in Speech-centric ML

 
 So far, there has been relatively little work done on audio-specific poisoning attacks. The first targeted attack against hybrid ASR systems is the VenoMave algorithm by [ 3 ] . For each x i x_{i} frame selected by the adversary for misclassification, that attacker constructs poisoned frames that ‘surround’ x i x_{i} in feature space and assigns them the target label. Thus any linear model that correctly classifies the poisoned frames will also misclassify the target frame. [ 71 ] proposed a clean-label attack to protect user’s speech data from use in downstream learning tasks such as speaker recognition or speech command recognition. This was accomplished by introducing perturbations that maximize the distance between the MFCC features of the clean and poisoned signal while simultaneously minimizing the perturbation’s magnitude. Thus any model trained on the poisoned data that uses MFCCs is likely to perform unreliably.

 
 
 

#### 4.2.3 Defenses against Poisoning Attacks in Speech-centric ML

 
 Many of the defenses proposed for poisoning attacks attempt to identify and remove the poisoned samples from the training data. For example, [ 179 ] extracted learned representation for all training samples with a given label. They then compute a singular value decomposition (SVD) and use the right singular vectors to compute outlier scores for each sample. Samples with a high outlier score are removed, and the model is retrained. Model activations were also used to detect poisoned samples by [ 34 ] , but the authors employ a clustering approach in place of the SVD. They also suggest fine-tuning the model on the re-labeled poisoned data rather than training from scratch on the filtered dataset. [ 69 ] leveraged the fact the sample contains a trigger that should be consistently classified as that target class even under perturbation. By randomly combining different images, the check for samples that exhibit low entropy in the distribution of predicted labels. Some defensive method focus on addressing backdoors in the model itself. For example, [ 189 ] tested for backdoors by computing the minimal perturbations required to change the classification result of all samples to a given target label. If a perturbation of a relatively small magnitude exists, then this may indicate the presence of a backdoor. The ‘Fine-pruning’ defense presented by [ 107 ] first pruned neurons that were inactive on clean samples under the hypothesis that these extraneous neurons can be co-opted by the backdoor trigger. Then the network was fine-tuned on a clean (un-poisoned) dataset to avoid overall performance degradation. [ 169 ] provided a method to upper bound the effects of a potential poisoning attack in the case where a data sanitation defense was used to remove outliers before training.

 
 
 
 
 

## Chapter 5 Privacy in Speech-centric Machine Learning

 
 Despite the promises modern speech applications can deliver, they also raise significant concerns and risks, such as exposing sensitive information that people might wish to keep confidential. The sensitive information can be individual attributes (e.g., age, gender), states (e.g., health, emotions), or biometric fingerprints. This section presents a comprehensive review of the privacy and security challenges related to trustworthy speech processing.

 
 

### 5.1 Taxonomies of Privacy-related Speech-centric ML

 
 In this subsection, we create a taxonomy of existing privacy-related topics on speech-centric ML in Figure 5.1 . In the categorization shown in Figure 5.1 , we organize papers based on privacy threats, mitigation mechanisms, downstream applications, and training paradigms. In the first place, we categorize privacy threats based on privacy attacks, including Property Inference Attacks (PIA), Membership Inference Attacks (MIA), Identity Inference Attacks (IIA), and Content Inference Attacks (CIA). For example, in PIA, the privacy attacker can obtain or infer speaker attributes like demographic information. On the other hand, we can divide privacy-related literature based on privacy mitigation methods. Specifically, the literature surrounding privacy-preserving speech processing can be categorized into algorithmic solutions and hardware solutions. Moreover, the most studied downstream speech applications related to privacy topics can be categorized into automatic speaker verification, keyword spotting, automatic speech recognition, and speech emotion recognition. Lastly, we discuss the privacy-related speech-centric ML based on the training paradigm: centralized learning and federated learning (FL).

 
 
 
 
 
 Privacy-enhancing 
 Speech-centric Machine Learning 
 
 
 
 Privacy Threats 
 
 
 
 Property Inference Attack 
 (PIA) 
 
 
 
 Membership Inference Attack 
 (MIA) 
 
 
 
 Identity Inference Attack 
 (IIA) 
 
 
 
 Content Inference Attack 
 (CIA) 
 
 
 
 Mitigation Mechanisms 
 
 
 
 Algorithmic Solution 
 
 
 
 Hardware Solution 
 
 
 
 Downstream Applications 
 
 
 
 Automatic Speech Recognition 
 (ASR)) 
 
 
 
 Speech Emotion Recognition 
 (SER) 
 
 
 
 Keyword Spotting 
 
 
 
 Speaker Verification 
 
 
 
 Training Paradigms 
 
 
 
 Centralized Learning 
 
 
 
 Federated Learning 
 
 
 
 [ 62 ] 
 
 
 
 [ 65 ] 
 
 
 
 [ 94 ] 
 
 
 
 [ 170 ] 
 
 
 
 [ 127 ] 
 
 
 
 [ 8 ] 
 
 
 
 [ 7 ] 
 
 
 
 [ 183 ] 
 
 
 
 [ 124 ] 
 
 
 
 [ 157 ] 
 
 
 
 [ 82 ] 
 
 
 
 [ 138 ] 
 
 
 
 [ 176 ] 
 
 
 
 [ 140 ] 
 
 
 
 [ 168 ] 
 
 
 
 [ 77 ] 
 
 
 
 [ 116 ] 
 
 
 
 [ 132 ] 
 
 
 
 [ 143 ] 
 
 
 
 [ 170 ] 
 
 
 
 [ 4 ] 
 
 
 
 [ 11 ] 
 
 
 
 [ 109 ] 
 
 
 
 [ 89 ] 
 
 
 
 [ 94 ] 
 
 
 
 [ 205 ] 
 
 
 
 [ 174 ] 
 
 
 
 [ 102 ] 
 
 
 
 [ 84 ] 
 
 
 
 [ 138 ] 
 
 
 
 [ 177 ] 
 
 
 
 [ 176 ] 
 
 
 
 Rahulama 
 thavan et al. 2018 
 
 
 
 [ 25 ] 
 
 
 
 [ 100 ] 
 
 
 
 [ 19 ] 
 
 
 
 [ 11 ] 
 
 
 
 [ 140 ] 
 
 
 
 [ 89 ] 
 
 
 
 [ 40 ] 
 
 
 
 [ 175 ] 
 
 
 
 [ 207 ] 
 
 
 
 [ 11 ] 
 
 
 
 [ 9 ] 
 
 
 
 [ 167 ] 
 
 
 
 [ 25 ] 
 
 
 
 [ 62 ] 
 
 
 
 [ 65 ] 
 
 
 
 [ 94 ] 
 
 
 
 [ 14 ] 
 
 
 
 [ 184 ] 
 
 
 
 [ 66 ] 
 
 
 
 [ 53 ] 
 
 
 
 [ 205 ] 
 
 
 
 [ 58 ] 
 
 
 
 [ 111 ] 
 
 
 
 [ 102 ] 
 
 
 
 [ 85 ] 
 
 
 
 [ 84 ] 
 
 
 
 [ 133 ] 
 
 
 
 [ 138 ] 
 
 
 
 [ 77 ] 
 
 
 
 Rahulama 
 thavan et al. 2018 
 
 
 
 [ 180 ] 
 
 
 
 [ 162 ] 
 
 
 
 [ 41 ] 
 
 
 
 [ 11 ] 
 
 
 
 [ 62 ] 
 
 
 
 [ 94 ] 
 
 
 
 [ 207 ] 
 
 
 
 [ 89 ] 
 
 
 
 [ 183 ] 
 
 
 
 [ 170 ] 
 
 
 
 [ 9 ] 
 
 
 
 [ 82 ] 
 
 
 
 [ 184 ] 
 
 
 
 [ 67 ] 
 
 
 
 [ 204 ] 
 
 
 
 [ 102 ] 
 
 
 
 [ 200 ] 
 
 
 
 [ 77 ] 
 
 
 
 [ 207 ] 
 
 
 
 [ 47 ] 
 
 
 
 [ 84 ] 
 
 
 Figure 5.1 : Taxonomy of privacy-related speech processing and modeling works. 
 
 
 Figure 5.2 : Overview of the privacy threats in speech-centric models. 
 
 
 

### 5.2 Privacy Threats

 
 In a privacy attack, the goal of an adversary is to acquire information that was not intended to be disclosed, including speech content, speaker demographics, and voice fingerprint. There are four mainstream privacy attacks against speech-centric applications as demonstrated in Figure 5.2 : Property Inference Attacks (PIA), Membership Inference Attacks (MIA), Identity Inference Attacks (IIA), and Content Inference Attacks (CIA). We provide a brief overview of each privacy attack and highlight works that either identified new privacy risks or were the first to investigate privacy threats in their respective domains.

 
 
 Figure 5.3 : Overview of the Membership inference attack. The privacy attacker first trains a set of shadow models with shadow training datasets. Once the shadow training is finished, the attacker gathers the model outputs from the target model and the shadow model to train the MIA classifier. The MIA classifier infers the membership property given an input posterior from the target model. 
 
 
 Property Inference Attacks (PIA) : PIA occurs when the adversary attempts to infer private attributes which are unrelated to the primary learning task. A notable example of the PIA in speech-centric applications is to infer the gender attribute using a pre-trained gender classification model, while the target application is to classify emotions or transcribe text. Here, the speech data that is accessible to the attacker can either be the raw speech recordings or processed speech features like MFCCs. In addition to gender property, adversaries can perform classification to predict the age ( [ 154 ] ), the language used ( [ 112 ] ), or even the health status ( [ 6 ] ) of the speaker from the speech data. However, since most existing speech-related datasets only include the annotations of gender but not other properties, the majority of the PIA works in speech applications focus on gender classification. Furthermore, apart from using features derived from raw speech data, the adversaries could perform privacy attacks through training updates generated in the collaborative training process presented by [ 67 ] .

 
 
 Membership Inference Attacks (MIA) : Membership inference aims to determine the participation of a data instance in training the target model. The idea of the MIA was first proposed by [ 160 ] , where the author assumed the attacker could access posterior estimations of a data sample by querying the target model. The attacker then used the posterior distribution to speculate whether the query data was in the training data. The general framework of MIAs is presented in Figure 5.3 . In MIAs, the attacker typically starts with a training procedure called shadow training which emulates the target training procedure. To perform the shadow training, the attacker often gathers a collection of shadow training datasets that share similar data distribution or data format to the target training data. The attacker then trains a classifier to perform MIAs using a collection of posteriors from training and shadow data. Although MIAs have been widely studied in computer vision and natural language processing, there is a limited amount of work in speech modeling. In speech-centric ML, [ 157 ] were the first to investigate MIAs on ASR models and their results show that the attacker can infer membership of the speech data with a moderate precision score. Furthermore, it is important to point out that the success rate of MIAs depends largely on the speakers, while some speakers are more vulnerable to MIAs. However, the connection between the attack success rate of MIAs and the speaker remains to be determined. Recently, [ 183 ] designed the MIA against pre-trained speech models trained using self-supervised learning (SSL) in black-box settings. Their results indicate that both utterance and speaker-level MIAs are feasible against SSL-based speech models.

 
 
 Identity Inference Attacks (IIA) : The IIA against speech-centric applications is a class of privacy attacks where adversaries can extract personally identifiable information (PII) from speech data for re-identification or impersonation purposes. However, we highlight that such identifiers often have applications in speaker identification and automatic speaker verification tasks. Traditional speaker identification frameworks rely on the extraction of i-vector from MFCCs ( [ 50 ] ). i-vector is a low-dimensional fixed-length representation of a speech utterance extracted using a data-driven approach. The state-of-art speaker identification systems in more recent years involve the extraction of the x-vector ( [ 164 ] ), which is the embedding extracted from Deep Neural Networks. Many papers ( [ 82 , 138 , 175 , 140 , 167 , 132 , 116 ] ) have proposed various privacy mitigation methods to prevent speaker re-identification. The vast majority of these works thus far applied adversarial training to disentangle the unique speaker information embedded in the speech recordings.

 
 
 Content Inference Attacks (CIA) : In addition to speaker properties, speaker memberships, and biometric identifiers, the textual contents of the speech can expose significant privacy concerns. For example, speech recordings often contain the name, address, and contact information of the speaker or the people around the speaker. Moreover, speech recorded in sensitive settings like business meetings can include proprietary information. Therefore, the content inference is to extract textual content from speech recordings. The naive approach to performing the content inference is through ASR itself. More recently, researchers have discovered that untended memorization in training deep speech models can also leak training data records ( [ 11 , 109 ] ). Specifically, the attacker can deliberately synthesize speech contents by guessing content patterns in the training data. For example, the crafted speech utterance can be "_ lives in the 1st street." where _ can be the silence audio snippet. The idea of the attack is that the ASR model would output the name along with the phrase "lives in the 1st street.", like "Bob lives in the 1st street.", as a consequence of model memorization.

 
 
 

### 5.3 Mitigation Mechanisms

 
 In this paragraph, we continue our review of privacy-related literature on speech-centric ML based on mitigation strategies. We split the related papers into algorithmic and hardware solutions based on such criteria.

 
 

#### 5.3.1 Algorithmic Solutions

 
 There is a large body of work falling into algorithmic solutions. More concretely, differential privacy (DP), adversarial training, and encryption are frequently adopted mitigation methods in speech-centric applications.

 
 
 Differential Privacy (DP) 

 
 The idea of DP was first introduced by [ 55 ] . Essentially, the DP is a rigorous privacy definition that guarantees the exclusion or the inclusion of any particular data record in the dataset has a negligible impact on the original data distribution. In other words, the privacy attacker cannot distinguish the data distribution changes by including or excluding any particular data point in the original dataset. Mathematically speaking, we can define DP given privacy parameters ϵ \epsilon and δ \delta shown below:

 
 
 Definition 5.3.1 ( ( ϵ , δ ) (\epsilon,\delta) -DP) . 
 
 A random mechanism ℳ \mathcal{M} satisfies ( ϵ , δ ) (\epsilon,\delta) -LDP, where ϵ 0 \epsilon 0 and δ ∈ [ 0 , 1 ) \delta\in[0,1) , if and only if for any two adjacent data sets 𝒟 \mathcal{D} and 𝒟 ′ \mathcal{D^{\prime}} in universe 𝒳 \mathcal{X} , we have: 

 
 
 

 
 | 
 P ​ r ​ ( ℳ ⁡ ( 𝒟 ) ) ≤ e ϵ ​ P ​ r ​ ( ℳ ⁡ ( 𝒟 ′ ) ) + δ Pr(\mathcal{M}(\mathcal{D}))\leq e^{\epsilon}Pr(\mathcal{M}(\mathcal{D^{\prime}}))+\delta | 
 | 
 (5.1) | 
 

 
 
 
 Here, ϵ 0 \epsilon 0 is defined as the privacy budget in DP, and a lower ϵ \epsilon represents stronger privacy protection ( [ 57 ] ). Specifically, ϵ \epsilon provides the bound of all outputs on neighboring data sets 𝒟 \mathcal{D} and 𝒟 ′ \mathcal{D^{\prime}} , which differ by one sample in a database. The typical way to achieve differential privacy is through noise perturbation. When considering centralized speech applications, [ 142 ] proposed VoiceMask that conceals the voiceprints of the speaker using differential privacy. Unlike the DP implementation in VoiceMask, Preech ( [ 4 ] ) ensured DP by adding dummy words in the output transcripts. Besides the literature on centralized training, we have observed an increasing trend of DP research in Federated Learning. For example, [ 67 ] applied the DP to mitigate property inference attackers in training the FL-based speech emotion recognition model. Sotto Voce ( [ 159 ] ) was another recently proposed work that explored DP in Federated speech recognition.

 
 
 
 Adversarial Training 

 
 As [ 126 ] pointed out, adversarial training is based upon information-theoretic privacy. The fundamental theory behind adversarial training is mutual information. For example, given a sensitive property or unique identifier z z and the associated speech data 𝐱 \mathbf{x} , we want to learn the perturbation h ⁡ ( ⋅ ) h(\cdot) that maximizes the mutual information I ⁡ ( h ⁡ ( 𝐱 ) , z ) {I(h(\mathbf{x});z)} . This learning objective is typically turned into the following adversarial training objectives as suggested by [ 165 ] :

 
 
 

 
 | 
 min ψ ⁡ max ϕ ⁡ ℒ ⁡ ( a ​ d ​ v ψ ​ ( h ϕ ​ ( 𝐱 ) ) , z ) \min_{\mathbf{\psi}}\max_{\mathbf{\phi}}\;\mathcal{L}(adv_{\mathbf{\psi}}(h_{\phi}(\mathbf{x})),z) | 
 | 
 (5.2) | 
 

 
 
 Essentially, we would like to train an adversary that is able to infer z z accurately, and meanwhile, we aim to improve the quality of the perturbation that confuses the adversary classifier. This training objective is normally combined with the target training objective in the learning phase. As we described in PIAs, many papers used adversarial training to disentangle the gender attribute from the speech signal. [ 94 ] were the first ones to propose to use of adversarial training to remove gender property in the SER task. Following [ 94 ] , [ 62 ] combined adversarial training with feature selection to greatly reduce the gender inference risks in SER. Another common approach to conducting adversarial training is through generating adversarial examples. For instance, in ASR training, [ 170 ] presented a generative adversarial network that fed gender-ambiguous training samples to train the ASR model. This design attempted to disentangle gender from training utterances. Aside from unlearning demographics from speech signals, [ 40 ] tried to unlearn PII such as x-vector by synthesizing speech utterances from a large pool of x-vector.

 
 
 
 Encryption 

 
 The last popular privacy-defending mechanism in this category is encryption. Many papers exploit Homomorphic Encryption (HE) for secure training of speech-centric models. [ 58 ] and [ 205 ] investigated the use of HE in keyword spotting systems, and [ 53 ] were the first to investigate HE in SER applications. On top of HE integrations, [ 138 ] adapted secure multiparty computation (SMC) protocols that substantially reduce the computation overhead needed to satisfy the privacy constraints. However, in our literature search, we cannot find related papers investigating HE in the ASR system. The lack of ASR research in this direction can be caused by the heavy computation required in encryption computation. Meanwhile, the modern ASR system demands a significant amount of computing resources.

 
 
 
 

#### 5.3.2 Hardware Solutions

 
 Compared with algorithmic mitigation in speech-centric applications, fewer research papers work on hardware solutions. Based on the sampled literature, we divide the hardware solution into sensing strategies and trust computing environments. In the context of audio sensing, [ 100 ] aimed to improve the privacy of audio data recorded from wearables by audio subsample and audio shredding on the device. Specifically, audio shredding was to randomize the sequence of recorded audio features. These two sensing strategies promised to provide useful audio features that were secure from reconstruction attacks. Additionally, [ 64 ] introduced a wearable audio solution that enhances privacy through sampling low-level acoustic characteristics instead of raw audio samples to study workplace stress ( [ 128 , 199 ] ). On the other hand, there has been a growing interest in recent years in using a trusted computing environment (TEE) in speech-centric applications. For example, VoiceGuard by [ 25 ] was one of the first works demonstrating the use of Intel SGX, a widely available TEE implementation, on the ASR application. Last but not least, [ 19 ] built a TEE architecture called Offline Model Guard (OMG) that allowed running KWS tasks on the pre-dominant mobile computing platform ARM.

 
 
 
 

### 5.4 Downstream Speech Applications

 
 Here, we review the privacy-related speech literature based on the downstream applications. In this review, we select popular speech applications, including automatic speech recognition (ASR), speech emotion recognition (SER), automatic speaker verification (ASV), and keyword spotting (KWS).

 
 
 ASR : As shown in Figure 5.1 , ASR is the most studied speech application in the context of privacy. These works implement privacy-enhancing features by removing gender property ( [ 8 ] ), biometric identifiers ( [ 140 ] ), and sensitive content ( [ 11 ] ) from speech signals. Specifically, the scientific community has also introduced VoicePrivacy 2020 ( [ 177 ] ) and VoicePrivacy 2022 challenges ( [ 176 ] ), with the target to evaluate privacy-preserving ASR modeling frameworks that suppress biometric identifiers in the speech signal. Currently, most of the privacy-enhancing ASR systems are proposed in the centralized setting, and Federated ASR training remains a challenge in speech-centric ML research. In this survey, we find only a few presented works ( [ 207 , 200 ] ) that focus on ASR modeling using federated learning. As [ 200 ] concludes, ASR models suffer significant utility loss using Federated learning due to the nature of the complexity and high variability residing in speech data. The lack of ASR modeling works in the FL domain can also be caused by the expensive computation requirements of the ASR models. Unlike the algorithmic solutions to reduce privacy risks, [ 25 ] introduced the VoiceGuard architecture that protects user privacy using inside a trusted execution environment (TEE). Last, we also want to highlight that the Librispeech ( [ 136 ] ) dataset is commonly used in privacy-related ASR works.

 
 
 SER : In our review, we identify that many papers ( [ 94 , 65 , 62 ] ) in this domain focus on disentangling the gender attribute from the speech signal using adversarial training. Among all these papers, [ 94 ] was the only work that considers multi-modal learning with other modalities. As opposed to gender obfuscation works mentioned above, [ 14 ] was the first to explore speaker anonymization using siamese neural network architecture. Apart from centralized speech emotion recognition, many recent works ( [ 66 , 184 , 67 ] ) explored privacy risks in federated learning settings. Specifically, the IEMOCAP dataset ( [ 28 ] ) is one of the most used datasets in conducting these experiments.

 
 
 KWS : The literature of privacy-preserving keyword spotting systems ( [ 205 , 58 ] ) are mainly based on homomorphic encryption (HE) solutions. However, as we discussed earlier, this method has major constraints in computation efficiency and is extremely challenging to deploy in the field. Alternatively, DataMix by [ 111 ] improves privacy by generating data samples through the mixup approach ( [ 203 ] ). The mixup data is a mixture of data samples that can effectively prevent privacy attacks like IIA while preserving the target model utility.

 
 
 ASV : Interestingly, most privacy-centered speech modeling papers treat speaker identity as sensitive information. Most of these papers heavily studied the obfuscation of the speaker identity, or in other words, reduce the performance of the speaker verification system, where the target application is often ASR, SER, or other speech-related applications. Since the i-vector or the x-vector already carries the biometric identifier of the speaker, many works that attempt to preserve privacy in the speaker verification systems focus on redesigning the system using homomorphic encryption solutions ( [ 162 , 138 ] ). However, the homomorphic encryption frameworks require heavy computations and are frequently impractical for real-life deployment. In contrast to these holomorphic proposals, [ 146 ] designed a randomization algorithm that significantly reduced the computation overhead for privacy-enhancing speaker verification using the i-vector. With the popularity of Federated Learning in more recent years, [ 77 ] investigated the on-device learning schemes for local speaker verification.

 
 
 

### 5.5 Training Paradigms

 

#### 5.5.1 Centralized

 
 Centralized learning requires collecting the raw speech data. In this setting, the speech signal is normally sampled at the client device and is then transferred to the service provider’s server for post-processing. The collection of speech data often draws substantial privacy concerns as speech signals encapsulate demographics, health information, PII, or sensitive speech content. If service providers are untrusted, they may not only infer the mentioned private information from speech data but also even render a person identifiable information.

 
 
 To decrease the privacy risk of gathering speech signals, many privacy regulations have been issued in recent years, such as European Union’s General Data Protection Regulation (GDPR) law ( [ 187 ] ). Noticeably, GDPR does not explicitly consider the machine learning models as personal data, but recent works imply that ML models themselves could be covered by GDPR as models may memorize sensitive information during the training process. With regard to the research community, a large amount of effort has been made to decrease the privacy risks in centralized speech systems like ASR, SER, and ASV. Many of these papers, as summarized earlier, focus on perturbing the speech data or intermediate speech features to disentangle private information. Widely used speech data perturbation methods are differential privacy, adversarial training, or generative methods. We would also stress that most current works emphasize hiding/removing PII, but fewer works studied the topic of preventing the rendering of PII.

 
 
 

#### 5.5.2 Federated Learning

 
 Section 5.5.1 has introduced the plethora of privacy risks centralized computing poses among which a significant attack concept is PII being transferred to a centralized server, wherein it is prone to cybersecurity attacks or curious server attacks. As an alternative to the traditional methods of training machine learning on a single server, federated learning (FL) employs a server-client model such that the data never leaves the client. Instead, the objective of the server is to aggregate all the model updates from the clients. Unlike centralized training, the clients train the model on a local dataset and transmit the updated parameters instead of the raw data. The general learning procedure of FL was shown in Figure 5.4 .

 
 
 Figure 5.4 : The training process of Federated Learning. 
 
 
 FL Optimization techniques The earliest optimization algorithm proposed to ensure convergence of the global model was done by [ 118 ] , known as Federated Averaging ( FedAvg ). FedAvg is similar to SGD, wherein the client model performs multiple iterations of updates before communicating the updates to the server. Although FedAvg has been shown to have great success, it has convergence issues in heterogeneous data settings, i.e., when the data distribution is non-IID and in cases wherein a limited number of clients participate in every global update. One method proposed to improve optimization for heterogeneous data is the Stochastic Controlled Averaging Algorithm by [ 97 ] (SCAFFOLD) by using control variates which reduce the client drift away from the global optima. More recently, [ 15 ] proposed three adaptive optimization algorithms, FedAdaGard, FedAdam, and FedYOGI, which are the FL versions of AdaGard, Adam, and YOGI, respectively. From empirical estimates, FedOpt outperforms most federated optimization strategies. Federated learning for speech processing tasks has been explored for numerous applications, mainly in keyword spotting, automatic speech recognition, and speech emotion recognition. Specifically, [ 204 ] provided a comprehensive benchmark for various audio-related tasks.

 
 
 KWS With the ubiquitous usage of smart assistants such as Siri, Google Voice, and Alexa, keyword spotting is an essential downstream speech processing task. Furthermore, it requires a relatively low parameter count ( ∼ \sim 200k) to achieve state-of-the-art performance, making it ideal for federated learning. [ 102 , 85 ] proposed the FL approach for keyword spotting tasks. Both papers provided models with comparable performance to a centrally trained model on their respective datasets. [ 85 ] observed that the performance of the FL model depends on the strategies employed to deal with the non-IID data. For example, data augmentation through SpecAug ( [ 137 ] ) or the usage of adaptive optimization methods such as ADAM in the local clients have been observed to be effective in preserving performance. In more recent work, [ 84 ] demonstrated a real-time on-device training of an FL model for Keyword Spotting. Also, it introduces semi-supervised learning in an FL scenario and self-correcting labels based on metadata during the recordings.

 
 
 ASR 
 [ 54 ] was one of the earliest works to propose FL for Speech Recognition tasks. In order to train with heterogeneous data, a two-level hierarchical optimization strategy was proposed, which involved a local client optimization followed by a global optimization and retraining of the global model on the client side held out dataset. This helps the training process better adjust to client drift. In addition, a weight model averaging is proposed, which helps improve the convergence speed. In order to account for data heterogeneity, the addition of variational noise has been proposed by [ 80 ] wherein each client model is added with a local random variational vector. The author named this method as federated variational noise (FVN). FVN has been shown to improve the relative performance of the federated learning models in a non-IID data scenario.

 
 
 With the constraints on the computational costs and the model sizes, there has been a focus on employing a cross-silo FL framework. Since the number of clients is smaller and each client has much larger computing power than a cross-device setting, usage of large-scale ASR models is justifiable. [ 46 ] proposed a cross-silo FL framework that contained a detailed analysis of training an ASR system with multi-domain data, i.e., each client has a specific domain exclusive data such as read speech, conversational data, meeting data etc. It introduces the client-adaptive federated learning (CAFT) method which accounts for the differing domain modalities across clients and adapts the client’s data using a transform. [ 131 ] experimented a modification to FedAvg termed as FedAvg-DS wherein DS stands for Diversity scaling. This modification emphasizes accounting for the variability in gradient directions of the local client updates.

 
 
 Recent efforts have enabled cross-device FL-based ASR training by [ 81 ] , which employed federated dropout ( [ 30 ] ) in order to reduce the model sizes in the clients and at the same time obtain a fully trained ASR at the server side. Additionally, it has been shown that training with federated dropout allows sub-models of the fully trained model to have comparable performance allowing for deployment on devices with varying computing capacities. [ 197 ] introduced partial variable training (PVT) which involves freezing layers in clients and training a specific set of layers per client and aggregating them per layer for the server updates.

 
 
 SER 
In order to deploy these models for real-time usage, we have to note that the availability of labeled data to the clients is extremely low, if not non-existent. Hence, semi-supervised FL frameworks are increasingly popular to train SER models with limited labeled data points per training client and a larger unlabelled set of data. The unlabelled sets of data are used in the supervised training by predicting pseudo labels. [ 184 ] generated the pseudo labels for the unlabelled models and retains them based on the confidence measure of the label. Whereas [ 66 ] used multiview pseudo-labeling ( [ 193 ] ) coupled with uncertainty-aware pseudo-labeling selection process ( [ 149 ] ) to generate the pseudo labels.

 
 
 Figure 5.5 : Challenges in Federated speech-centric applications. 
 
 
 

#### 5.5.3 Challenges in FL

 
 Despite considerable improvements in privacy owing to the transition from a centralized to a decentralized training approach in FL, significant challenges remain for the ubiquitous adoption of FL models in speech processing. Although some speech processing tasks can attain the state of the art performance within parameter counts of 300k, tasks such as speech recognition and speech generation require about 100M parameters to obtain performance that is equivalent to state of art. However, clients in an FL are constrained by low computational power and model sizes. This provides an opportunity for future research into better optimization techniques or smaller models which can enable these tasks to be trained in a federated setting.

 
 
 Another issue that arises while training models at clients in a supervised manner is the lack of clean labels. Therefore, semi-supervised and unsupervised techniques are of interest since they do not have to deal with the lack of clean labeled data. Some methods to tackle this issue in current works include employing a student-teacher framework providing weak labels on the local datasets and using external metadata to infer the labels based on user actions.

 
 
 Moreover, despite user data not being transferred to the server, it has been shown that the gradients which are transferred are susceptible to a multitude of privacy attacks, such as data reconstruction attacks, property inference attacks, membership inference attacks, and poisoning attacks which adversely affect the privacy-preserving nature of FL. Although there have been numerous works on privacy attacks and defenses in federated learning, most of them focus on image or text tasks. Privacy attacks for speech processing tasks are a relatively under-explored area so are the defenses. However, it is an increasing field of interest owing to the large-scale adoption of voice assistants and their privacy concerns.

 
 
 

#### 5.5.4 Privacy Attacks in FL

 
 [ 175 ] proposed two attacks on ASR models in order to infer the speaker identity. One is purely statistical, and the other is a neural network-based attack to infer speaker identity based on the outputs of hidden layers from speaker models transmitted from the clients to the server. It has been demonstrated that this attack is successful with EER values obtained about 1-2%. Similar property inference attacks were performed on SER models by [ 63 ] , wherein the gender of the client was inferred from the model updates transmitted to the central server, which could either be the gradients or weight parameters. The attack is performed by shadow training ( [ 160 ] ) on an open dataset similar to the target dataset, followed by using the model updates obtained during the shadow training to infer the gender of the speaker. A point to note is that in both the previous attacks, it is observed that the first hidden layer of the model provides the most information about the speaker identity.

 
 
 

#### 5.5.5 Defences for FL

 
 Defenses for FL models mainly include the use of DP ( [ 55 ] ), or Homomorphic Encryption ( [ 135 ] ). In the speech processing domain, [ 67 ] deployed User-Level Differential Privacy (UDP) to mitigate property inference attacks from SER models. However, one limitation of the proposed defense is that when the adversary obtains multiple updates of the models, the performance of the defense degrades. [ 33 ] proposed a two-layered defense mechanism i.e., a combination of randomization and adversarial training in a federated setting to defend against FGSM, PGD, and Deepfool attacks for SER tasks.

 
 
 
 
 

## Chapter 6 Bias and Fairness in Speech-centric Machine Learning

 

### 6.1 Fairness in Machine Learning

 
 The advance of machine learning technology has led to its ubiquitous application spanning several domains including healthcare, travel, and also the judicial system. While ML can alleviate human effort and promote automation, it has to be used cautiously to avoid potential biases to infiltrate the automatic decision-making process ( [ 73 ] ). For example, a popular study investigated Recidivism Prediction Instruments (RPI) – ML technology used to predict if a person who had committed a criminal offense in the past is likely to commit an offense in the future, and found that the popular COMPAS RPI was biased against black defendants ( [ 12 ] ).

 
 
 Similarly, evidence of bias has been found in face recognition ( [ 166 , 150 ] ), natural language applications ( [ 113 ] ) as well as voice assistants ( [ 52 ] ). As explained below, such bias can originate from either the training data, features/model or incorrect application of algorithm. Hence, mitigation methods have been proposed at each of these stages of the pipeline, from data curation to model deployment. In this section, we briefly delineate individual and group fairness, introduce different causes of bias, and finally delve into mitigation methods proposed in the speech domain.

 
 

#### 6.1.1 Notions of Fairness:

 
 The fairness of machine learning algorithms is measured along one of two dimensions:

 
 
 a) Individual fairness: tracks fairness/bias at the level of an individual member of a population, with the assumption being that similar individuals will be treated similarly ( [ 56 , 101 ] ).

 
 
 b) Group fairness: measures relative bias between different subgroup populations of interest ( [ 86 , 21 ] ).
These sub-groups are typically divided along the lines of sensitive attributes of individuals such as gender, race, or age.

 
 
 

#### 6.1.2 Causes of Bias:

 
 Unfair machine learning applications can be largely attributed to one of two main causes:

 
 
 a) Data Bias :
Sources of data bias can be broadly categorized into measurement bias, omitted variable bias, representation bias, sampling bias and aggregation bias ( [ 120 ] ).
These biases creep in either due to a) incorrect sampling of data from different subgroups, b) biased feature representations used for modeling, or c) misinterpretation of population statistics for subgroup statistics.
A common mechanism used to mitigate data bias due to imbalance in subgroups is to over-sample data from minority subgroups during training ( [ 52 , 68 ] ).

 
 
 b) Algorithmic Bias :
Algorithmic bias arises when an ML algorithm introduces bias in the system or amplifies existing bias in the data.
Such bias permeates in the absence of carefully designed ML algorithms.
Algorithmic bias can be classified based on training data bias, bias in designing an algorithm or bias in deploying an algorithm ( [ 48 ] ).
Training data bias can originate from an algorithm that propagates existing bias in the data ( [ 166 ] ).
Bias in the design of an algorithm can be either a focus bias, wherein the algorithm uses features that are biased towards specific sub-groups, or processing bias, wherein the algorithm itself introduces bias as in the case of a statistically biased estimator.
Finally, bias can occur in the deployment or interpretation of an algorithm.
For example, using an algorithm outside the context it is developed for can lead to unfair results.
Similarly, using incorrect performance metrics that do not reflect the distribution of the data can lead to misinterpretation of the results.
Methods used to mitigate algorithmic bias include adversarial training and joint multi-task training ( [ 172 , 182 , 139 ] ).

 
 
 Different formulations have been proposed based on the fairness objective being targeted ( [ 186 ] ).
For example, some fairness metrics such as statistical parity and equal acceptance rate only consider the predicted outcome of a model.
More commonly used metrics, including equalized odds, predictive equality, predictive parity and equal opportunity further consider the true label in their fairness definition.

 
 
 In speech processing applications, however, most work still use standard performance metrics to obtain fairness metrics (e.g., word error rate (WER) for ASR, equal error rate (EER) for ASV)).
As shown by [ 139 ] , these metrics do not always hold, especially when applied to subgroups of population.
In the following subsection, we outline different fairness related work including mitigation strategies in speech processing applications and present the different data resources that have been curated for fairness research.

 
 
 Table 6.1 : Speech datasets for fairness evaluation 
 
 
 
 
 Dataset | 
 Task | 
 
 
 
 Sensitive attributes | 

 
 (# classes) | 

 | 
 No. Hours | 

 
 
 
 Fairvoice ( [ 68 ] ) | 
 ASV | 
 Gender (2), Age (9), Language (6) | 
 1700 | 

 
 Casual Conversations ( [ 105 ] ) | 
 ASR | 
 Gender (2), Skin Type (6) | 
 572 | 

 
 TedTalk ( [ 2 ] ) | 
 ASR | 
 Gender (3), Race (4) | 
 564 | 

 
 Artie Bias Corpus ( [ 123 ] ) | 
 ASR | 
 Gender (3), Age (8), Accent (3) | 
 2.4 | 

 
 Speech Accents ( [ 192 ] ) | 
 ASR | 
 Accents (7), Gender (2), Age | 
 1 | 

 

 
 
 
 
 

### 6.2 Fairness in Speech-centric Applications

 
 Compared with the flourishing of fairness research in other domains, such as facial analysis and natural language processing, the importance of addressing bias issues in speech processing has been underestimated for a long time. Up to today, there is still a limited amount of studies focusing on speech-related fairness. However, the accuracy degradation of a biased speech model on a specific demographic group not only leads to users’ inconvenience but also makes the group believe the product is not designed for them. Therefore, it is essential to systematically evaluate the speech-related models’ performance on fairness and investigate effective methods to mitigate the bias in speech models.
Several datasets have been proposed to advance fairness research in the speech domain (See Table 6.1 . In addition to the task-specific labels (speaker/transcript for ASV/ASR), these datasets include labels for sensitive demographic attributes such as gender, race, age, accent, and language.

 
 

#### 6.2.1 Fairness in Automatic Speaker Verification

 
 ASV systems suffer from bias in different demographic attributes, such as gender, age, language, and ethnic. For example, [ 171 ] analyzed the performance difference of statistical speaker models regarding gender and age. However, the analysis is based on the verification scores while lacking deeper insights into the bias of the decision model and the distribution of speaker embeddings. As discussed earlier, these two are essential factors in clearly uncovering bias in ML models.
More recently, [ 139 ] investigated bias in ASV, showing the importance of distribution scores and also proposed fairness metrics for ASV instead of standard EER. They also explored mitigation strategies of adversarial training and multi-task training to reduce gender bias in ASV systems.

 
 
 In addition to gender and age, ASV systems have also been found to be biased across languages. A study of UN-meetings ( [ 88 ] ) investigated the effect of language in speaker verification results and found that ASV degraded for Russian language as compared to English.
 [ 96 ] forced the ASV learner to focus on poorly performing instances by weighting samples with an adversarial reweighting network and demonstrated that the reweighting method significantly improves the performance of ASV across different subgroups of gender and nationality.

 
 
 NIST introduced new languages in their Speaker Recognition Evaluation protocol in 2016 ( [ 153 ] ) and 2018 ( [ 152 ] ), to investigate the influence of language in the speaker verification system. In the 2021 VoxCeleb Speaker Recognition Challenge (VoxSRC) ( [ 26 ] ), the language attribute was added to the speaker verification track, aiming to encourage researchers to solve the performance degradation issue in the multi-lingual setting and boost the fairness in speaker verification. It is worth noting that the majority of the Fairness studies on the ASV task provide the evaluations using the VoxCeleb dataset series ( [ 130 , 42 ] ).
 [ 37 ] explored the bias in speaker identification systems across different race groups and found that latinxs performed significantly worse than Caucasian speakers.
Recently, [ 68 ] explored the fairness in deep learning-based ASV systems by collecting a speaker dataset – Fairvoice, conducting performance analysis on EER and score (FAR, FRR) distributions, and providing more understanding of how diverse speaker verification is correlated with demographic attributes.
To standardize the ASV fairness evaluation, [ 178 ] devised a framework to provide comprehensive fairness evaluation metrics and visualization methods to present model’s fairness across subgroups.

 
 
 

#### 6.2.2 Fairness in Automatic Speech Recognition

 
 In the past decade, due to the development of deep learning and the availability of large-scale speech and language databases, the word error rate (WER) of ASR models has decreased to a satisfactory level in many languages. However, the fairness of ASR systems still raises the interests of researchers from psychological, sociology, and engineering backgrounds ( [ 122 , 147 ] ). [ 122 ] investigated the ASR failure on African American Vernacular English and demonstrated the detrimental impact on African American users from the psychological perspective. In addition to proposing a set of methodologies to model the users’ feelings and experience in fairness research, Mengesha et al. also encourages researchers to spot more light on fairness AI. [ 147 ] introduced an automated testing framework (AEQUEVOX) for evaluating the fairness of ASR systems. By conducting extensive fairness experiments on four datasets with three commercial ASRs, [ 147 ] validated the ASR fairness violation on non-native English, female, and Nigerian English speakers.
With the recent advent of self-supervised learning, approaches such as wav2vec2 ( [ 16 ] ) are being widely used in a variety of speech recognition related tasks. [ 24 ] investigated the impact of pretrained data distribution on the fairness performance across subgroups. By pretraining the wav2vec 2.0 with gender-specific and different proportion of gender data, it is demonstrated that the fairness is related to downstream integration and balanced-gender pretraining data does not necessarily reduce bias.

 
 
 To mitigate the bias in ASR on the groups across geographic locations and demographic attributes, [ 52 ] proposed an initial method, oversampling under minority groups and undersampling majority groups, to reduce the performance gap between different cohorts. [ 105 ] presented results from multiple ASR models on the Casual Conversations dataset and observed the significant WER difference across gender and ethic. To accurately evaluate the ASR fairness issue on racial demographics, [ 110 ] adopted mixed-effects Poisson regression to mitigate the negative influence from nuisance factors, such as speaker, context, phoneme, prosody, etc.

 
 
 

#### 6.2.3 Fairness in Speech Emotion Recognition

 
 Fairness in SER is important across multiple areas because the performance differences resulting from gender and race are significant in most of emotion recognition scenarios ( [ 158 ] ). [ 76 ] investigated gender-based bias in speech emotion recognition and mitigated unwanted bias through adversarial training and additional weight for the objective function. In recent years, the large-scale self-supervised learning (SSL) model has become popular in computer vision, language processing, and audio processing. With the widespread use of SSL, [ 188 ] have shone light on the performance variance and demonstrated that transformer-based SSL models have moderate fairness scores in the SER area.

 
 
 
 
 

## Chapter 7 Future Directions

 
 In this section, we discuss the main challenges and potential research opportunities for speech-centric trustworthy machine learning in order to inspire readers to research this field further.

 
 

### 7.1 Privacy

 
 In recent years, self-supervised learning speech models such as Wav2Vec 2.0 ( [ 16 ] ) and Whisper ( [ 145 ] ) have established the SOTA performance for many downstream speech tasks such as ASR. However, privacy-related topics, such as MIAs, on these emerging model techniques have not been explored extensively. As we discussed earlier, [ 183 ] was the only one to investigate MIA threats to SSL speech models. Their results imply that the latent speech representation of SSL models holds the membership information of the input speech signals. However, they only perform some preliminary defenses on MIAs and findings from these defenses are limited. Therefore, it is valuable and critical to extend the work presented in [ 183 ] to broader SSL speech models, downstream speech tasks, and MIA mitigation strategies.

 
 
 In addition, there exist more property inferences where current PIAs have not been explored but are of demand in speech-centric ML, e.g., age, health status, etc. As our review points out, the majority of literature focuses on gender obfuscation, while none of the studies attempts to generalize the existing approaches to demographics like age or race. Apart from the lack of exploration of broader demographics, many studies only evaluate their works on several clean-audio benchmarks like Librispeech ( [ 136 ] ) and IEMOCAP ( [ 28 ] ), while the robustness and efficacy of many proposed privacy-enhancing approaches on the more dynamic recording conditions are unknown.

 
 
 As we also presented in the Privacy review section, Federated Learning has become an emerging research topic in almost every field of ML. Nevertheless, compared to NLP and CV domains, the FL on speech-centric tasks stays largely unexplored. One of the significant challenges is to enable the training of the ASR system in the FL setting. Due to the nature of the ASR modeling, expensive computations are typically mandatory, while most mobile devices cannot afford to train and run these heavy computing models. Therefore, it is urgent and essential to investigate efficient FL training approaches for ASR models. Last but not least, most literature to date focuses on dealing with missing labels and decoupling heterogeneity conditions in FL, and fewer works are targeting privacy risks in federated speech learning. As a result, an exciting but critical research direction is systematically studying the privacy risks in federated speech modeling.

 
 
 

### 7.2 Safety

 
 While consideration has been given to speech-specific evasion attacks, there has been far less work on audio-modality defenses. Speech signals possess unique structural and temporal properties that distinguish them from images. Traditional defense methods from the computer vision domain fail to fully leverage these attributes in aid of adversarial robustness. Furthermore, there have been some works such as by [ 144 ] and [ 195 ] that consider ‘over the air attacks’ in which the adversarial audio travels to the model by means of an acoustic channel (as opposed to being directly fed to the model input). More work is needed in this area to gauge the feasibility and threat level of evasion attacks in this more realistic setting. Finally, there is relatively little work on poisoning attacks and defenses for speech systems. Given that large-scale datasets are increasingly procured from unverified sources (i.e. internet posts, client data in FL), it is critical that we better understand the potential risks and how to mitigate them.

 
 
 

### 7.3 Fairness

 
 As outlined in the previous section, research on fairness and bias mitigation in the speech domain is limited to a few preliminary works.
These are typically restricted to fairness evaluations on small attribute-balanced datasets for either gender or accent.
With the introduction of newer datasets (Table 6.1 ), we hope to see more fairness studies along the demographic attributes of ethnicity, language, etc. as well as intersectional attributes (e.g, female Spanish vs male Spanish).
Furthermore, there has been little work exploring the extent of bias in recent SOTA models such as Wav2Vec2.0 ( [ 16 ] ) and Whisper ( [ 145 ] ) for both ASR and ASV.
There has also been little to no work in other speech tasks such as SER.
Finally, the need for fairness-specific metrics is highlighted by [ 139 ] , with most of the existing literature using standard metrics such as EER and WER.

 
 
 

### 7.4 Balance between Fairness, Privacy, and Safety

 
 Despite the tremendous effort in designing trustworthy machine learning techniques over the last decade, most of these applications attempt to isolate one single aspect of trustworthiness in their modeling work. Consequently, a limited amount of work explores the impact of mitigating one trustworthy risk over other trustworthy challenges. Recently, [ 32 ] demonstrated that privacy and fairness are often opposed to each other in trustworthy machine learning. The author showed that increasing model fairness requires optimizing objectives that typically constrain the model to perform equally on every subgroup, leading the model to memorize training samples from the unprivileged subgroups. Hence, enhancing model fairness frequently raises significant privacy concerns in exposing the private information of the training data.

 
 
 Apart from investigating the relationship between fairness and privacy, [ 194 ] found that safety-awareness learning poses a disparate impact on the fairness risk of subgroups. The author also proposed a Fair-Robust-Learning (FRL) framework to enhance the model fairness while performing adversarial defenses. However, both of these works solely investigate computer vision applications, while the interplay between privacy, fairness, and safety in speech-centric applications remains largely unexplored. It is, therefore, critical for future researchers to investigate the interactions between different trustworthiness aspects in speech-centric applications.

 
 
 

### 7.5 Trustworthy Multimodal Applications

 
 Besides speech-centric applications that we discussed in this article, speech signals have broader usage cases in diverse multimodal applications, such as multimedia applications, audio-visual understanding, and multimodal sentiment analysis ( [ 95 ] ). Typically, multimodal applications are reported with a higher system performance than unimodal models. Consequently, one promising future research direction is to explore speech-based multimodal applications that attempt to achieve more satisfactory balances between fairness, privacy, and safety.

 
 
 
 

## Chapter 8 Conclusion

 
 As machine learning becomes more prevalent in speech-centric modeling, the scientific community has also become more aware of concerns and risks over the trustworthiness of these systems. This survey paper aims to provide a comprehensive and systematic summary of the recent efforts made to protect privacy, ensure fairness, and defend against adversarial attacks in these speech-centric systems. Increasingly, we identify several open problems of importance to address, such as investigating the extent of bias in recent SOTA models. Through this review, we hope to provide the necessary knowledge for future research in speech-centric trustworthy ML.

 
 
 

## Chapter 9 Acknowledgement

 
 This work was supported by DARPA (Grant No: HR00112020009) and USC Amazon Center for Secure and Trusted Machine Learning.

 
 
 

## References

 
 
 [1] 
 Hadi Abdullah et al.
 
 “Hear" no evil", see" kenansville"*: Efficient and transferable black-box attacks on speech recognition and voice identification systems”
 
 In 2021 IEEE Symposium on Security and Privacy (SP) , 2021, pp. 712–729
 
 IEEE
 

 
 [2] 
 Rupam Acharyya, Shouman Das, Ankani Chattoraj and Md. Tanveer
 
 “FairyTED: A Fair Rating Predictor for TED Talk Data”
 
 In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020 
 
 AAAI Press, 2020, pp. 338–345
 

 
 [3] 
 Hojjat Aghakhani et al.
 
 “VenoMave: Targeted Poisoning Against Speech Recognition”
 
 In arXiv preprint arXiv:2010.10682 , 2020
 

 
 [4] 
 Shimaa Ahmed, Amrita Chowdhury, Kassem Fawaz and Parmesh Ramanathan
 
 “Preech: A System for { \{ Privacy-Preserving } \} Speech Transcription”
 
 In 29th USENIX Security Symposium (USENIX Security 20) , 2020, pp. 2703–2720
 

 
 [5] 
 Mehmet Akçay and Kaya Oğuz
 
 “Speech emotion recognition: Emotional models, databases, features, preprocessing methods, supporting modalities, and classifiers”
 
 In Speech Communication 116 
 
 Elsevier, 2020, pp. 56–76
 

 
 [6] 
 Alican Akman et al.
 
 “Evaluating the covid-19 identification resnet (cider) on the interspeech covid-19 from audio challenges”
 
 In Frontiers in Digital Health 4 
 
 Frontiers Media SA, 2022
 

 
 [7] 
 Ranya Aloufi, Hamed Haddadi and David Boyle
 
 “Emotionless: Privacy-preserving speech analysis for voice assistants”
 
 In arXiv preprint, arXiv: 1908.03632 , 2019
 

 
 [8] 
 Ranya Aloufi, Hamed Haddadi and David Boyle
 
 “Privacy-preserving voice analysis via disentangled representations”
 
 In Proceedings of the 2020 ACM SIGSAC Conference on Cloud Computing Security Workshop , 2020, pp. 1–14
 

 
 [9] 
 Ranya Aloufi, Hamed Haddadi and David Boyle
 
 “Configurable Privacy-Preserving Automatic Speech Recognition”
 
 In Proc. Interspeech 2021 , 2021, pp. 861–865
 
 DOI: 10.21437/Interspeech.2021-1783 
 

 
 [10] 
 Moustafa Alzantot, Bharathan Balaji and Mani Srivastava
 
 “Did you hear that? adversarial examples against automatic speech recognition”
 
 In arXiv preprint arXiv:1801.00554 , 2018
 

 
 [11] 
 Ehsan Amid et al.
 
 “Extracting Targeted Training Data from ASR Models, and How to Mitigate It”
 
 In Proc. Interspeech 2022 , 2022, pp. 2803–2807
 
 DOI: 10.21437/Interspeech.2022-10895 
 

 
 [12] 
 Julia Angwin, Jeff Larson, Surya Mattu and Lauren Kirchner
 
 “Machine bias”
 
 In Ethics of Data and Analytics 
 
 Auerbach Publications, 2016, pp. 254–264
 

 
 [13] 
 Rosana Ardila et al.
 
 “Common voice: A massively-multilingual speech corpus”
 
 In arXiv preprint arXiv:1912.06670 , 2019
 

 
 [14] 
 Priya Arora and Theodora Chaspari
 
 “Exploring siamese neural network architectures for preserving speaker identity in speech emotion classification”
 
 In Proceedings of the 4th International Workshop on Multimodal Analyses Enabling Artificial Agents in Human-Machine Interaction , 2018, pp. 15–18
 

 
 [15] 
 Muhammad Asad, Ahmed Moustafa and Takayuki Ito
 
 “FedOpt: Towards communication efficiency and privacy preservation in federated learning”
 
 In Applied Sciences 10.8 
 
 MDPI, 2020, pp. 2864
 

 
 [16] 
 Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed and Michael Auli
 
 “wav2vec 2.0: A framework for self-supervised learning of speech representations”
 
 In Advances in Neural Information Processing Systems 33 , 2020, pp. 12449–12460
 

 
 [17] 
 Jon Barker, Shinji Watanabe, Emmanuel Vincent and Jan Trmal
 
 “The fifth ’CHiME’ speech separation and recognition challenge: dataset, task and baselines”
 
 In arXiv preprint arXiv:1803.10609 , 2018
 

 
 [18] 
 Marco Barreno, Blaine Nelson, Anthony Joseph and J Tygar
 
 “The security of machine learning”
 
 In Machine Learning 81.2 
 
 Springer, 2010, pp. 121–148
 

 
 [19] 
 Sebastian Bayerl et al.
 
 “Offline model guard: Secure and private ML on mobile devices”
 
 In 2020 Design, Automation Test in Europe Conference Exhibition (DATE) , 2020, pp. 460–465
 
 IEEE
 

 
 [20] 
 Yoshua Bengio, Yann Lecun and Geoffrey Hinton
 
 “Deep learning for AI”
 
 In Communications of the ACM 64.7 
 
 ACM New York, NY, USA, 2021, pp. 58–65
 

 
 [21] 
 Richard Berk et al.
 
 “Fairness in criminal justice risk assessments: The state of the art”
 
 In Sociological Methods Research 50.1 
 
 Sage Publications Sage CA: Los Angeles, CA, 2021, pp. 3–44
 

 
 [22] 
 Battista Biggio et al.
 
 “Evasion attacks against machine learning at test time”
 
 In Joint European conference on machine learning and knowledge discovery in databases , 2013, pp. 387–402
 
 Springer
 

 
 [23] 
 Battista Biggio, Blaine Nelson and Pavel Laskov
 
 “Poisoning attacks against support vector machines”
 
 In arXiv preprint arXiv:1206.6389 , 2012
 

 
 [24] 
 Marcely Boito, Laurent Besacier, Natalia. Tomashenko and Yannick Estève
 
 “A Study of Gender Impact in Self-supervised Models for Speech-to-Text Systems”
 
 In INTERSPEECH 
 
 ISCA, 2022, pp. 1278–1282
 

 
 [25] 
 Ferdinand Brasser et al.
 
 “VoiceGuard: Secure and Private Speech Processing”
 
 In Proc. Interspeech 2018 , 2018, pp. 1303–1307
 
 DOI: 10.21437/Interspeech.2018-2032 
 

 
 [26] 
 Andrew Brown et al.
 
 “VoxSRC 2021: The Third VoxCeleb Speaker Recognition Challenge”
 
 In CoRR abs/2201.04583 , 2022
 
 arXiv: https://arxiv.org/abs/2201.04583 
 

 
 [27] 
 Andrew Brown et al.
 
 “Playing a part: Speaker verification at the movies”
 
 In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2021, pp. 6174–6178
 
 IEEE
 

 
 [28] 
 Carlos Busso et al.
 
 “IEMOCAP: Interactive emotional dyadic motion capture database”
 
 In Language resources and evaluation 42.4 
 
 Springer, 2008, pp. 335–359
 

 
 [29] 
 Zhipeng Cai et al.
 
 “Generative adversarial networks: A survey toward private and secure applications”
 
 In ACM Computing Surveys (CSUR) 54.6 
 
 ACM New York, NY, USA, 2021, pp. 1–38
 

 
 [30] 
 Sebastian Caldas, Jakub Konečny, H McMahan and Ameet Talwalkar
 
 “Expanding the reach of federated learning by reducing client resource requirements”
 
 In arXiv preprint arXiv:1812.07210 , 2018
 

 
 [31] 
 Nicholas Carlini and David Wagner
 
 “Audio adversarial examples: Targeted attacks on speech-to-text”
 
 In 2018 IEEE security and privacy workshops (SPW) , 2018, pp. 1–7
 
 IEEE
 

 
 [32] 
 Hongyan Chang and Reza Shokri
 
 “On the privacy risks of algorithmic fairness”
 
 In 2021 IEEE European Symposium on Security and Privacy (EuroS P) , 2021, pp. 292–303
 
 IEEE
 

 
 [33] 
 Yi Chang et al.
 
 “Robust Federated Learning Against Adversarial Attacks for Speech Emotion Recognition”
 
 In arXiv preprint arXiv: 2203.04696 , 2022
 

 
 [34] 
 Bryant Chen et al.
 
 “Detecting backdoor attacks on deep neural networks by activation clustering”
 
 In arXiv preprint arXiv:1811.03728 , 2018
 

 
 [35] 
 Guangke Chen et al.
 
 “Who is real bob? adversarial attacks on speaker recognition systems”
 
 In 2021 IEEE Symposium on Security and Privacy (SP) , 2021, pp. 694–711
 
 IEEE
 

 
 [36] 
 Li-Wei Chen and Alexander Rudnicky
 
 “Exploring Wav2vec 2.0 fine-tuning for improved speech emotion recognition”
 
 In arXiv preprint arXiv:2110.06309 , 2021
 

 
 [37] 
 Xingyu Chen, Zhengxiong Li, Srirangaraj Setlur and Wenyao Xu
 
 “Exploring racial and gender disparities in voice biometrics”
 
 In Scientific Reports 12.1 
 
 Nature Publishing Group, 2022, pp. 1–12
 

 
 [38] 
 Xinyun Chen et al.
 
 “Targeted backdoor attacks on deep learning systems using data poisoning”
 
 In arXiv preprint arXiv:1712.05526 , 2017
 

 
 [39] 
 Peng Cheng and Utz Roedig
 
 “Personal voice assistant security and privacy—a survey”
 
 In Proceedings of the IEEE 110.4 
 
 IEEE, 2022, pp. 476–507
 

 
 [40] 
 Gopinath Chennupati et al.
 
 “ILASR: privacy-preserving incremental learning for automatic speech recognition at production scale”
 
 In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , 2022, pp. 2780–2788
 

 
 [41] 
 Oubaïda Chouchane et al.
 
 “Privacy-Preserving Voice Anti-Spoofing Using Secure Multi-Party Computation”
 
 In Proc. Interspeech 2021 , 2021, pp. 856–860
 
 DOI: 10.21437/Interspeech.2021-983 
 

 
 [42] 
 Joon Chung, Arsha Nagrani and Andrew Zisserman
 
 “Voxceleb2: Deep speaker recognition”
 
 In arXiv preprint arXiv:1806.05622 , 2018
 

 
 [43] 
 Moustapha Cisse et al.
 
 “Parseval networks: Improving robustness to adversarial examples”
 
 In International Conference on Machine Learning , 2017, pp. 854–863
 
 PMLR
 

 
 [44] 
 Leigh Clark et al.
 
 “The state of speech in HCI: Trends, themes and challenges”
 
 In Interacting with Computers 31.4 
 
 Oxford University Press, 2019, pp. 349–371
 

 
 [45] 
 Jeremy Cohen, Elan Rosenfeld and Zico Kolter
 
 “Certified adversarial robustness via randomized smoothing”
 
 In International Conference on Machine Learning , 2019, pp. 1310–1320
 
 PMLR
 

 
 [46] 
 Xiaodong Cui, Songtao Lu and Brian Kingsbury
 
 “Federated acoustic modeling for automatic speech recognition”
 
 In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2021, pp. 6748–6752
 
 IEEE
 

 
 [47] 
 Yue Cui et al.
 
 “Privacy-preserving Speech-based Depression Diagnosis via Federated Learning”
 
 In 2022 44th Annual International Conference of the IEEE Engineering in Medicine Biology Society (EMBC) , 2022, pp. 1371–1374
 
 IEEE
 

 
 [48] 
 David Danks and Alex London
 
 “Algorithmic Bias in Autonomous Systems.”
 
 In Ijcai 17 , 2017, pp. 4691–4697
 

 
 [49] 
 James Davidson et al.
 
 “The YouTube video recommendation system”
 
 In Proceedings of the fourth ACM conference on Recommender systems , 2010, pp. 293–296
 

 
 [50] 
 Najim Dehak et al.
 
 “Front-end factor analysis for speaker verification”
 
 In IEEE Transactions on Audio, Speech, and Language Processing 19.4 
 
 IEEE, 2010, pp. 788–798
 

 
 [51] 
 Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova
 
 “Bert: Pre-training of deep bidirectional transformers for language understanding”
 
 In arXiv preprint arXiv:1810.04805 , 2018
 

 
 [52] 
 Pranav Dheram et al.
 
 “Toward Fairness in Speech Recognition: Discovery and mitigation of performance disparities”
 
 In INTERSPEECH 
 
 ISCA, 2022, pp. 1268–1272
 

 
 [53] 
 Miguel Dias, Alberto Abad and Isabel Trancoso
 
 “Exploring hashing and cryptonet based approaches for privacy-preserving speech emotion recognition”
 
 In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2018, pp. 2057–2061
 
 IEEE
 

 
 [54] 
 Dimitrios Dimitriadis et al.
 
 “A Federated Approach in Training Acoustic Models.”
 
 In Interspeech , 2020, pp. 981–985
 

 
 [55] 
 Cynthia Dwork
 
 “Differential privacy: A survey of results”
 
 In International conference on theory and applications of models of computation , 2008, pp. 1–19
 
 Springer
 

 
 [56] 
 Cynthia Dwork et al.
 
 “Fairness through awareness”
 
 In Proceedings of the 3rd innovations in theoretical computer science conference , 2012, pp. 214–226
 

 
 [57] 
 Cynthia Dwork, Frank McSherry, Kobbi Nissim and Adam Smith
 
 “Calibrating noise to sensitivity in private data analysis”
 
 In Theory of cryptography conference , 2006, pp. 265–284
 
 Springer
 

 
 [58] 
 Daniel Elworth and Sunwoong Kim
 
 “HEKWS: Privacy-Preserving Convolutional Neural Network-based Keyword Spotting with a Ciphertext Packing Technique”
 
 In 2022 IEEE 24th International Workshop on Multimedia Signal Processing (MMSP) , 2022, pp. 01–06
 
 IEEE
 

 
 [59] 
 Mohammad Esmaeilpour, Patrick Cardinal and Alessandro Koerich
 
 “Class-conditional defense GAN against end-to-end speech attacks”
 
 In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2021, pp. 2565–2569
 
 IEEE
 

 
 [60] 
 Mohammad Esmaeilpour, Patrick Cardinal and Alessandro Koerich
 
 “Cyclic defense gan against speech adversarial attacks”
 
 In IEEE Signal Processing Letters 28 
 
 IEEE, 2021, pp. 1769–1773
 

 
 [61] 
 Florian Eyben, Martin Wöllmer and Björn Schuller
 
 “Opensmile: the munich versatile and fast open-source audio feature extractor”
 
 In Proceedings of the 18th ACM international conference on Multimedia , 2010, pp. 1459–1462
 

 
 [62] 
 Tiantian Feng, Hanieh Hashemi, Murali Annavaram and Shrikanth Narayanan
 
 “Enhancing Privacy Through Domain Adaptive Noise Injection For Speech Emotion Recognition”
 
 In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2022, pp. 7702–7706
 
 IEEE
 

 
 [63] 
 Tiantian Feng et al.
 
 “Attribute inference attack of speech emotion recognition in federated learning settings”
 
 In arXiv preprint arXiv:2112.13416 , 2021
 

 
 [64] 
 Tiantian Feng et al.
 
 “Tiles audio recorder: an unobtrusive wearable solution to track audio activity”
 
 In Proceedings of the 4th ACM Workshop on Wearable Systems and Applications , 2018, pp. 33–38
 

 
 [65] 
 Tiantian Feng and Shrikanth Narayanan
 
 “Privacy and Utility Preserving Data Transformation for Speech Emotion Recognition”
 
 In 2021 9th International Conference on Affective Computing and Intelligent Interaction (ACII) , 2021, pp. 1–7
 
 IEEE
 

 
 [66] 
 Tiantian Feng and Shrikanth Narayanan
 
 “Semi-FedSER: Semi-supervised Learning for Speech Emotion Recognition On Federated Learning using Multiview Pseudo-Labeling”
 
 In arXiv preprint arXiv:2203.08810 , 2022
 

 
 [67] 
 Tiantian Feng, Raghuveer Peri and Shrikanth Narayanan
 
 “User-Level Differential Privacy against Attribute Inference Attack of Speech Emotion Recognition on Federated Learning”
 
 In Proc. Interspeech 2022 , 2022, pp. 5055–5059
 
 DOI: 10.21437/Interspeech.2022-10060 
 

 
 [68] 
 Gianni Fenu, Hicham Lafhouli and Mirko Marras
 
 “Exploring algorithmic fairness in deep speaker verification”
 
 In International Conference on Computational Science and Its Applications , 2020, pp. 77–93
 
 Springer
 

 
 [69] 
 Yansong Gao et al.
 
 “Strip: A defense against trojan attacks on deep neural networks”
 
 In Proceedings of the 35th Annual Computer Security Applications Conference , 2019, pp. 113–125
 

 
 [70] 
 John Garofolo, David Graff, Doug Paul and David Pallett
 
 “Csr-i (wsj0) sennheiser ldc93s6b”
 
 In Web Download. Philadelphia: Linguistic Data Consortium , 1993
 

 
 [71] 
 Yunjie Ge et al.
 
 “WaveFuzz: A Clean-Label Poisoning Attack to Protect Your Voice”
 
 In arXiv preprint arXiv:2203.13497 , 2022
 

 
 [72] 
 Jonas Geiping et al.
 
 “Witches’ brew: Industrial scale data poisoning via gradient matching”
 
 In arXiv preprint arXiv:2009.02276 , 2020
 

 
 [73] 
 Benjamin van Giffen, Dennis Herhausen and Tobias Fahse
 
 “Overcoming the pitfalls and perils of algorithms: A classification of machine learning biases and mitigation methods”
 
 In Journal of Business Research 144 
 
 Elsevier, 2022, pp. 93–106
 

 
 [74] 
 Yuan Gong and Christian Poellabauer
 
 “Crafting adversarial examples for speech paralinguistics applications”
 
 In arXiv preprint arXiv:1711.03280 , 2017
 

 
 [75] 
 Ian Goodfellow, Jonathon Shlens and Christian Szegedy
 
 “Explaining and harnessing adversarial examples”
 
 In arXiv preprint arXiv:1412.6572 , 2014
 

 
 [76] 
 Cristina Gorrostieta et al.
 
 “Gender De-Biasing in Speech Emotion Recognition.”
 
 In INTERSPEECH , 2019, pp. 2823–2827
 

 
 [77] 
 Filip Granqvist et al.
 
 “Improving On-Device Speaker Verification Using Federated Learning with Privacy”
 
 In Proc. Interspeech 2020 , 2020, pp. 4328–4332
 
 DOI: 10.21437/Interspeech.2020-2944 
 

 
 [78] 
 Tianyu Gu, Brendan Dolan-Gavitt and Siddharth Garg
 
 “Badnets: Identifying vulnerabilities in the machine learning model supply chain”
 
 In arXiv preprint arXiv:1708.06733 , 2017
 

 
 [79] 
 Anmol Gulati et al.
 
 “Conformer: Convolution-augmented transformer for speech recognition”
 
 In arXiv preprint arXiv: 2005.08100 , 2020
 

 
 [80] 
 Dhruv Guliani, Françoise Beaufays and Giovanni Motta
 
 “Training speech recognition models with federated learning: A quality/cost framework”
 
 In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2021, pp. 3080–3084
 
 IEEE
 

 
 [81] 
 Dhruv Guliani et al.
 
 “Enabling on-device training of speech recognition models with federated dropout”
 
 In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2022, pp. 8757–8761
 
 IEEE
 

 
 [82] 
 Yaowei Han et al.
 
 “Voice- indistinguishability: Protecting voiceprint in privacy-preserving speech data release”
 
 In 2020 IEEE International Conference on Multimedia and Expo (ICME) , 2020, pp. 1–6
 
 IEEE
 

 
 [83] 
 Awni Hannun et al.
 
 “Deep speech: Scaling up end-to-end speech recognition”
 
 In arXiv preprint arXiv:1412.5567 , 2014
 

 
 [84] 
 Andrew Hard et al.
 
 “Production federated keyword spotting via distillation, filtering, and joint federated-centralized training”
 
 In Proc. Interspeech 2022 , 2022, pp. 76–80
 
 DOI: 10.21437/Interspeech.2022-11050 
 

 
 [85] 
 Andrew Hard et al.
 
 “Training keyword spotting models on non-iid data with federated learning”
 
 In arXiv preprint arXiv:2005.10406 , 2020
 

 
 [86] 
 Moritz Hardt, Eric Price and Nati Srebro
 
 “Equality of opportunity in supervised learning”
 
 In Advances in neural information processing systems 29 , 2016
 

 
 [87] 
 Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun
 
 “Deep residual learning for image recognition”
 
 In Proceedings of the IEEE conference on computer vision and pattern recognition , 2016, pp. 770–778
 

 
 [88] 
 Rajat Hebbar et al.
 
 “A Computational Tool to Study Vocal Participation of Women in UN-ITU Meetings”
 
 In 2021 International Conference on Content-Based Multimedia Indexing (CBMI) , 2021, pp. 1–4
 
 IEEE
 

 
 [89] 
 W. Huang, Steve Chien, Om Thakkar and Rajiv Mathews
 
 “Detecting Unintended Memorization in Language-Model-Fused ASR”
 
 In Proc. Interspeech 2022 , 2022, pp. 2808–2812
 
 DOI: 10.21437/Interspeech.2022-10909 
 

 
 [90] 
 Xiaowei Huang et al.
 
 “A survey of safety and trustworthiness of deep neural networks: Verification, testing, adversarial attack and defence, and interpretability”
 
 In Computer Science Review 37 
 
 Elsevier, 2020, pp. 100270
 

 
 [91] 
 Shehzeen Hussain et al.
 
 “ { \{ WaveGuard } \} : Understanding and Mitigating Audio Adversarial Examples”
 
 In 30th USENIX Security Symposium (USENIX Security 21) , 2021, pp. 2273–2290
 

 
 [92] 
 Ngoc Huynh et al.
 
 “Adversarial Attacks on Speech Recognition Systems for Mission-Critical Applications: A Survey”
 
 In arXiv preprint arXiv:2202.10594 , 2022
 

 
 [93] 
 Amna Irum and Ahmad Salman
 
 “Speaker verification using deep neural networks: A”
 
 In International Journal of Machine Learning and Computing 9.1 , 2019
 

 
 [94] 
 Mimansa Jaiswal and Emily Provost
 
 “Privacy enhanced multimodal neural representations for emotion recognition”
 
 In Proceedings of the AAAI Conference on Artificial Intelligence 34.05 , 2020, pp. 7985–7993
 

 
 [95] 
 Xingyu Jiang et al.
 
 “A review of multimodal image matching: Methods and applications”
 
 In Information Fusion 73 
 
 Elsevier, 2021, pp. 22–71
 

 
 [96] 
 Minho Jin et al.
 
 “Adversarial Reweighting for Speaker Verification Fairness”
 
 In INTERSPEECH 
 
 ISCA, 2022, pp. 4800–4804
 

 
 [97] 
 Sai Karimireddy et al.
 
 “Scaffold: Stochastic controlled averaging for federated learning”
 
 In International Conference on Machine Learning , 2020, pp. 5132–5143
 
 PMLR
 

 
 [98] 
 Shreya Khare, Rahul Aralikatte and Senthil Mani
 
 “Adversarial black-box attacks on automatic speech recognition systems using multi-objective evolutionary optimization”
 
 In arXiv preprint arXiv:1811.01312 , 2018
 

 
 [99] 
 Arne Köhn, Florian Stegen and Timo Baumann
 
 “Mining the spoken wikipedia for speech data and beyond”
 
 Fachbereich Informatik, 2016
 

 
 [100] 
 Sumeet Kumar et al.
 
 “Sound shredding: Privacy preserved audio sensing”
 
 In Proceedings of the 16th international workshop on mobile computing systems and applications , 2015, pp. 135–140
 

 
 [101] 
 Matt Kusner, Joshua Loftus, Chris Russell and Ricardo Silva
 
 “Counterfactual fairness”
 
 In Advances in neural information processing systems 30 , 2017
 

 
 [102] 
 David Leroy et al.
 
 “Federated learning for keyword spotting”
 
 In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2019, pp. 6341–6345
 
 IEEE
 

 
 [103] 
 Fangzhou Liao et al.
 
 “Defense against adversarial attacks using high-level representation guided denoiser”
 
 In Proceedings of the IEEE conference on computer vision and pattern recognition , 2018, pp. 1778–1787
 

 
 [104] 
 Bo Liu et al.
 
 “When machine learning meets privacy: A survey and outlook”
 
 In ACM Computing Surveys (CSUR) 54.2 
 
 ACM New York, NY, USA, 2021, pp. 1–36
 

 
 [105] 
 Chunxi Liu et al.
 
 “Towards Measuring Fairness in Speech Recognition: Casual Conversations Dataset Transcriptions”
 
 In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2022, Virtual and Singapore, 23-27 May 2022 
 
 IEEE, 2022, pp. 6162–6166
 
 DOI: 10.1109/ICASSP43922.2022.9747501 
 

 
 [106] 
 Haochen Liu et al.
 
 “Trustworthy ai: A computational perspective”
 
 In ACM Transactions on Intelligent Systems and Technology 14.1 
 
 ACM New York, NY, 2022, pp. 1–59
 

 
 [107] 
 Kang Liu, Brendan Dolan-Gavitt and Siddharth Garg
 
 “Fine-pruning: Defending against backdooring attacks on deep neural networks”
 
 In International Symposium on Research in Attacks, Intrusions, and Defenses , 2018, pp. 273–294
 
 Springer
 

 
 [108] 
 Yingqi Liu et al.
 
 “Trojaning attack on neural networks”, 2017
 

 
 [109] 
 Yuchen Liu, Apu Kapadia and Donald Williamson
 
 “Preventing sensitive-word recognition using self-supervised learning to preserve user-privacy for automatic speech recognition”
 
 In Proc. Interspeech 2022 , 2022, pp. 4207–4211
 
 DOI: 10.21437/Interspeech.2022-85 
 

 
 [110] 
 Zhe Liu, Irina-Elena Veliche and Fuchun Peng
 
 “Model-Based Approach for Measuring the Fairness in ASR”
 
 In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2022, Virtual and Singapore, 23-27 May 2022 
 
 IEEE, 2022, pp. 6532–6536
 

 
 [111] 
 Zhijian Liu et al.
 
 “Datamix: Efficient privacy-preserving edge-cloud inference”
 
 In European Conference on Computer Vision , 2020, pp. 578–595
 
 Springer
 

 
 [112] 
 Ignacio Lopez-Moreno et al.
 
 “Automatic language identification using deep neural networks”
 
 In 2014 IEEE international conference on acoustics, speech and signal processing (ICASSP) , 2014, pp. 5337–5341
 
 IEEE
 

 
 [113] 
 Kaiji Lu et al.
 
 “Gender bias in neural natural language processing”
 
 In Logic, Language, and Security 
 
 Springer, 2020, pp. 189–202
 

 
 [114] 
 Aleksander Madry et al.
 
 “Towards deep learning models resistant to adversarial attacks”
 
 In arXiv preprint arXiv:1706.06083 , 2017
 

 
 [115] 
 Mishaim Malik, Muhammad Malik, Khawar Mehmood and Imran Makhdoom
 
 “Automatic speech recognition: a survey”
 
 In Multimedia Tools and Applications 80.6 
 
 Springer, 2021, pp. 9411–9457
 

 
 [116] 
 Mohamed Maouche et al.
 
 “A Comparative Study of Speech Anonymization Metrics”
 
 In Proc. Interspeech 2020 , 2020, pp. 1708–1712
 
 DOI: 10.21437/Interspeech.2020-2248 
 

 
 [117] 
 Luz Martinez-Lucas, Mohammed Abdelwahab and Carlos Busso
 
 “The MSP-conversation corpus”
 
 In Interspeech 2020 , 2020
 

 
 [118] 
 Brendan McMahan et al.
 
 “Communication-efficient learning of deep networks from decentralized data”
 
 In Artificial intelligence and statistics , 2017, pp. 1273–1282
 
 PMLR
 

 
 [119] 
 Nicholas Mehlman, Anirudh Sreeram, Raghuveer Peri and Shrikanth Narayanan
 
 “Mel frequency spectral domain defenses against adversarial attacks on speech recognition systems”
 
 In JASA Express Letters 3.3 
 
 Acoustical Society of America, 2023, pp. 035208
 

 
 [120] 
 Ninareh Mehrabi et al.
 
 “A survey on bias and fairness in machine learning”
 
 In ACM Computing Surveys (CSUR) 54.6 
 
 ACM New York, NY, USA, 2021, pp. 1–35
 

 
 [121] 
 Shike Mei and Xiaojin Zhu
 
 “Using machine teaching to identify optimal training-set attacks on machine learners”
 
 In Twenty-Ninth AAAI Conference on Artificial Intelligence , 2015
 

 
 [122] 
 Zion Mengesha et al.
 
 “"I don’t Think These Devices are Very Culturally Sensitive." - Impact of Automated Speech Recognition Errors on African Americans”
 
 In Frontiers Artif. Intell. 4 , 2021, pp. 725911
 

 
 [123] 
 Josh Meyer, Lindy Rauchenstein, Joshua Eisenberg and Nicholas Howell
 
 “Artie bias corpus: An open dataset for detecting demographic bias in speech applications”
 
 In Proceedings of the 12th language resources and evaluation conference , 2020, pp. 6462–6468
 

 
 [124] 
 Yuantian Miao et al.
 
 “The audio auditor: user-level membership inference in Internet of Things voice services”
 
 In arXiv preprint arXiv:1905.07082 , 2019
 

 
 [125] 
 Riccardo Miotto et al.
 
 “Deep learning for healthcare: review, opportunities and challenges”
 
 In Briefings in bioinformatics 19.6 
 
 Oxford University Press, 2018, pp. 1236–1246
 

 
 [126] 
 Fatemehsadat Mireshghallah et al.
 
 “Privacy in deep learning: A survey”
 
 In arXiv preprint arXiv:2004.12254 , 2020
 

 
 [127] 
 Nicolas Müller, Franziska Diekmann and Jennifer Williams
 
 “Attacker Attribution of Audio Deepfakes”
 
 In Proc. Interspeech 2022 , 2022, pp. 2788–2792
 
 DOI: 10.21437/Interspeech.2022-129 
 

 
 [128] 
 Karel Mundnich et al.
 
 “TILES-2018, a longitudinal physiologic and behavioral data set of hospital workers”
 
 In Scientific Data 7.1 
 
 Nature Publishing Group UK London, 2020, pp. 354
 

 
 [129] 
 Luis Muñoz-Gonzãlez et al.
 
 “Towards poisoning of deep learning algorithms with back-gradient optimization”
 
 In Proceedings of the 10th ACM workshop on artificial intelligence and security , 2017, pp. 27–38
 

 
 [130] 
 Arsha Nagrani, Joon Chung and Andrew Zisserman
 
 “Voxceleb: a large-scale speaker identification dataset”
 
 In arXiv preprint arXiv:1706.08612 , 2017
 

 
 [131] 
 Kishore Nandury, Anand Mohan and Frederick Weber
 
 “Cross-silo federated training in the cloud with diversity scaling and semi-supervised learning”
 
 In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2021, pp. 3085–3089
 
 IEEE
 

 
 [132] 
 Alexandru Nelus et al.
 
 “Privacy-Preserving Siamese Feature Extraction for Gender Recognition versus Speaker Identification”
 
 In Proc. Interspeech 2019 , 2019, pp. 3705–3709
 
 DOI: 10.21437/Interspeech.2019-1148 
 

 
 [133] 
 Paul-Gauthier Noé et al.
 
 “Adversarial Disentanglement of Speaker Representation for Attribute-Driven Privacy Preservation”
 
 In Proc. Interspeech 2021 , 2021, pp. 1902–1906
 
 DOI: 10.21437/Interspeech.2021-1712 
 

 
 [134] 
 Margarita Osadchy et al.
 
 “No bot expects the DeepCAPTCHA! Introducing immutable adversarial examples, with applications to CAPTCHA generation”
 
 In IEEE Transactions on Information Forensics and Security 12.11 
 
 IEEE, 2017, pp. 2640–2653
 

 
 [135] 
 Pascal Paillier
 
 “Public-key cryptosystems based on composite degree residuosity classes”
 
 In International conference on the theory and applications of cryptographic techniques , 1999, pp. 223–238
 
 Springer
 

 
 [136] 
 Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur
 
 “Librispeech: an asr corpus based on public domain audio books”
 
 In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP) , 2015, pp. 5206–5210
 
 IEEE
 

 
 [137] 
 Daniel Park et al.
 
 “Specaugment: A simple data augmentation method for automatic speech recognition”
 
 In arXiv preprint arXiv:1904.08779 , 2019
 

 
 [138] 
 Manas Pathak and Bhiksha Raj
 
 “Privacy-preserving speaker verification and identification using gaussian mixture models”
 
 In IEEE Transactions on Audio, Speech, and Language Processing 21.2 
 
 IEEE, 2012, pp. 397–406
 

 
 [139] 
 Raghuveer Peri, Krishna Somandepalli and Shrikanth Narayanan
 
 “To train or not to train adversarially: A study of bias mitigation strategies for speaker recognition”
 
 In arXiv preprint arXiv:2203.09122 , 2022
 

 
 [140] 
 Champion Pierre, Anthony Larcher and Denis Jouvet
 
 “Are disentangled representations all you need to build speaker anonymization systems?”
 
 In Proc. Interspeech 2022 , 2022, pp. 2793–2797
 
 DOI: 10.21437/Interspeech.2022-10586 
 

 
 [141] 
 Soujanya Poria et al.
 
 “Meld: A multimodal multi-party dataset for emotion recognition in conversations”
 
 In arXiv preprint arXiv:1810.02508 , 2018
 

 
 [142] 
 Jianwei Qian et al.
 
 “Hidebehind: Enjoy voice input with voiceprint unclonability and anonymity”
 
 In Proceedings of the 16th ACM Conference on Embedded Networked Sensor Systems , 2018, pp. 82–94
 

 
 [143] 
 Jianwei Qian et al.
 
 “Towards privacy-preserving speech data publishing”
 
 In IEEE INFOCOM 2018-IEEE Conference on Computer Communications , 2018, pp. 1079–1087
 
 IEEE
 

 
 [144] 
 Yao Qin et al.
 
 “Imperceptible, robust, and targeted adversarial examples for automatic speech recognition”
 
 In International conference on machine learning , 2019, pp. 5231–5240
 
 PMLR
 

 
 [145] 
 Alec Radford et al.
 
 “Robust speech recognition via large-scale weak supervision”
 
 In OpenAI Blog , 2022
 

 
 [146] 
 Yogachandran Rahulamathavan et al.
 
 “Privacy-preserving iVector-based speaker verification”
 
 In IEEE/ACM Transactions on Audio, Speech, and Language Processing 27.3 
 
 IEEE, 2018, pp. 496–506
 

 
 [147] 
 Sai Rajan, Sakshi Udeshi and Sudipta Chattopadhyay
 
 “AequeVox: Automated Fairness Testing of Speech Recognition Systems”
 
 In Fundamental Approaches to Software Engineering - 25th International Conference, FASE 2022, Held as Part of the European Joint Conferences on Theory and Practice of Software, ETAPS 2022, Munich, Germany, April 2-7, 2022, Proceedings 13241 , Lecture Notes in Computer Science
 
 Springer, 2022, pp. 245–267
 

 
 [148] 
 Douglas Reynolds, Thomas Quatieri and Robert Dunn
 
 “Speaker verification using adapted Gaussian mixture models”
 
 In Digital signal processing 10.1-3 
 
 Elsevier, 2000, pp. 19–41
 

 
 [149] 
 Mamshad Rizve, Kevin Duarte, Yogesh Rawat and Mubarak Shah
 
 “In defense of pseudo-labeling: An uncertainty-aware pseudo-label selection framework for semi-supervised learning”
 
 In arXiv preprint arXiv: 2101.06329 , 2021
 

 
 [150] 
 Joseph Robinson et al.
 
 “Face recognition: too bias, or not too bias?”
 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops , 2020, pp. 0–1
 

 
 [151] 
 Anthony Rousseau, Paul Deléglise and Yannick Esteve
 
 “TED-LIUM: an Automatic Speech Recognition dedicated corpus.”
 
 In LREC , 2012, pp. 125–129
 

 
 [152] 
 Seyed Sadjadi et al.
 
 “The 2018 NIST Speaker Recognition Evaluation”
 
 In Interspeech 2019, 20th Annual Conference of the International Speech Communication Association, Graz, Austria, 15-19 September 2019 
 
 ISCA, 2019, pp. 1483–1487
 

 
 [153] 
 Seyed Sadjadi et al.
 
 “The 2016 NIST Speaker Recognition Evaluation”
 
 In Interspeech 2017, 18th Annual Conference of the International Speech Communication Association, Stockholm, Sweden, August 20-24, 2017 
 
 ISCA, 2017, pp. 1353–1357
 

 
 [154] 
 Saeid Safavi, Martin Russell and Peter Jančovič
 
 “Automatic speaker, age-group and gender identification from children’s speech”
 
 In Computer Speech Language 50 
 
 Elsevier, 2018, pp. 141–156
 

 
 [155] 
 Lea Schönherr et al.
 
 “Adversarial attacks against automatic speech recognition systems via psychoacoustic hiding”
 
 In arXiv preprint arXiv:1808.05665 , 2018
 

 
 [156] 
 Ali Shafahi et al.
 
 “Poison frogs! targeted clean-label poisoning attacks on neural networks”
 
 In Advances in neural information processing systems 31 , 2018
 

 
 [157] 
 Muhammad. Shah et al.
 
 “Evaluating the Vulnerability of End-to-End Automatic Speech Recognition Models to Membership Inference Attacks”
 
 In Proc. Interspeech 2021 , 2021, pp. 891–895
 
 DOI: 10.21437/Interspeech.2021-1188 
 

 
 [158] 
 Garima Sharma and Abhinav Dhall
 
 “A survey on automatic multimodal emotion recognition in the wild”
 
 In Advances in Data Science: Methodologies and Applications 
 
 Springer, 2021, pp. 35–64
 

 
 [159] 
 Michael Shoemate et al.
 
 “Sotto Voce: Federated Speech Recognition with Differential Privacy Guarantees”
 
 In arXiv preprint arXiv: 2207.07816 , 2022
 

 
 [160] 
 Reza Shokri, Marco Stronati, Congzheng Song and Vitaly Shmatikov
 
 “Membership inference attacks against machine learning models”
 
 In 2017 IEEE symposium on security and privacy (SP) , 2017, pp. 3–18
 
 IEEE
 

 
 [161] 
 David Silver et al.
 
 “Mastering the game of Go with deep neural networks and tree search”
 
 In nature 529.7587 
 
 Nature Publishing Group, 2016, pp. 484–489
 

 
 [162] 
 Paris Smaragdis and Madhusudana Shashanka
 
 “A framework for secure speech recognition”
 
 In IEEE Transactions on Audio, Speech, and Language Processing 15.4 
 
 IEEE, 2007, pp. 1404–1413
 

 
 [163] 
 Nathalie Smuha
 
 “The EU approach to ethics guidelines for trustworthy artificial intelligence”
 
 In Computer Law Review International 20.4 
 
 Verlag Dr. Otto Schmidt, 2019, pp. 97–106
 

 
 [164] 
 David Snyder et al.
 
 “X-vectors: Robust dnn embeddings for speaker recognition”
 
 In 2018 IEEE international conference on acoustics, speech and signal processing (ICASSP) , 2018, pp. 5329–5333
 
 IEEE
 

 
 [165] 
 Jiaming Song et al.
 
 “Learning controllable fair representations”
 
 In The 22nd International Conference on Artificial Intelligence and Statistics , 2019, pp. 2164–2173
 
 PMLR
 

 
 [166] 
 Nisha Srinivas et al.
 
 “Face recognition algorithm bias: Performance differences on images of children and adults”
 
 In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops , 2019, pp. 0–4
 

 
 [167] 
 Brij Srivastava, Aurélien Bellet, Marc Tommasi and Emmanuel Vincent
 
 “Privacy-Preserving Adversarial Representation Learning in ASR: Reality or Illusion?”
 
 In Proc. Interspeech 2019 , 2019, pp. 3700–3704
 
 DOI: 10.21437/Interspeech.2019-2415 
 

 
 [168] 
 Brij Srivastava et al.
 
 “Design Choices for X-Vector Based Speaker Anonymization”
 
 In Proc. Interspeech 2020 , 2020, pp. 1713–1717
 
 DOI: 10.21437/Interspeech.2020-2692 
 

 
 [169] 
 Jacob Steinhardt, Pang Koh and Percy Liang
 
 “Certified defenses for data poisoning attacks”
 
 In Advances in neural information processing systems 30 , 2017
 

 
 [170] 
 Dimitrios Stoidis and Andrea Cavallaro
 
 “Generating gender-ambiguous voices for privacy-preserving speech recognition”
 
 In Interspeech 2022, 23rd Annual Conference of the International Speech Communication Association, Incheon, Korea, 18-22 September 2022 
 
 ISCA, 2022
 
 DOI: 10.21437/Interspeech.2022-11322 
 

 
 [171] 
 Lara Stoll
 
 “Finding Difficult Speakers in Automatic Speaker Recognition”, 2011
 

 
 [172] 
 Sining Sun et al.
 
 “Domain adversarial training for accented speech recognition”
 
 In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2018, pp. 4854–4858
 
 IEEE
 

 
 [173] 
 Christian Szegedy et al.
 
 “Intriguing properties of neural networks”
 
 In arXiv preprint arXiv:1312.6199 , 2013
 

 
 [174] 
 Thomas Thebaud, Gaël Le and Anthony Larcher
 
 “Spoofing Speaker Verification With Voice Style Transfer And Reconstruction Loss”
 
 In 2021 IEEE International Workshop on Information Forensics and Security (WIFS) , 2021, pp. 1–7
 
 IEEE
 

 
 [175] 
 Natalia Tomashenko et al.
 
 “Privacy attacks for automatic speech recognition acoustic models in a federated learning framework”
 
 In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2022, pp. 6972–6976
 
 IEEE
 

 
 [176] 
 Natalia Tomashenko et al.
 
 “The VoicePrivacy 2022 Challenge Evaluation Plan”
 
 In arXiv preprint arXiv:2203.12468 , 2022
 

 
 [177] 
 Natalia Tomashenko et al.
 
 “The VoicePrivacy 2020 Challenge: Results and findings”
 
 In Computer Speech Language 74 
 
 Elsevier, 2022, pp. 101362
 

 
 [178] 
 Wiebke Toussaint and Aaron Ding
 
 “SVEva Fair: A Framework for Evaluating Fairness in Speaker Verification”
 
 In CoRR abs/2107.12049 , 2021
 

 
 [179] 
 Brandon Tran, Jerry Li and Aleksander Madry
 
 “Spectral signatures in backdoor attacks”
 
 In Advances in neural information processing systems 31 , 2018
 

 
 [180] 
 Amos Treiber et al.
 
 “Privacy-preserving PLDA speaker verification using outsourced secure computation”
 
 In Speech Communication 114 
 
 Elsevier, 2019, pp. 60–71
 

 
 [181] 
 Edmondo Trentin and Marco Gori
 
 “A survey of hybrid ANN/HMM models for automatic speech recognition”
 
 In Neurocomputing 37.1-4 
 
 Elsevier, 2001, pp. 91–126
 

 
 [182] 
 Aditay Tripathi, Aanchan Mohan, Saket Anand and Maneesh Singh
 
 “Adversarial learning of raw speech features for domain invariant speech recognition”
 
 In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2018, pp. 5959–5963
 
 IEEE
 

 
 [183] 
 Wei-Cheng Tseng, Wei-Tsung Kao and Hung-yi Lee
 
 “Membership Inference Attacks Against Self-supervised Speech Models”
 
 In Proc. Interspeech 2022 , 2022, pp. 5040–5044
 
 DOI: 10.21437/Interspeech.2022-11245 
 

 
 [184] 
 Vasileios Tsouvalas, Tanir Ozcelebi and Nirvana Meratnia
 
 “Privacy-preserving Speech Emotion Recognition through Semi-Supervised Federated Learning”
 
 In 2022 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops) , 2022, pp. 359–364
 
 IEEE
 

 
 [185] 
 Ashish Vaswani et al.
 
 “Attention is all you need”
 
 In Advances in neural information processing systems 30 , 2017
 

 
 [186] 
 Sahil Verma and Julia Rubin
 
 “Fairness definitions explained”
 
 In 2018 ieee/acm international workshop on software fairness (fairware) , 2018, pp. 1–7
 
 IEEE
 

 
 [187] 
 Paul Voigt and Axel Von
 
 “The eu general data protection regulation (gdpr)”
 
 In A Practical Guide, 1st Ed., Cham: Springer International Publishing 10.3152676 
 
 Springer, 2017, pp. 10–5555
 

 
 [188] 
 Johannes Wagner et al.
 
 “Dawn of the transformer era in speech emotion recognition: closing the valence gap”
 
 In CoRR abs/2203.07378 , 2022
 

 
 [189] 
 Bolun Wang et al.
 
 “Neural cleanse: Identifying and mitigating backdoor attacks in neural networks”
 
 In 2019 IEEE Symposium on Security and Privacy (SP) , 2019, pp. 707–723
 
 IEEE
 

 
 [190] 
 Changhan Wang et al.
 
 “Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation”
 
 In arXiv preprint arXiv: 2101.00390 , 2021
 

 
 [191] 
 Pete Warden
 
 “Speech commands: A dataset for limited-vocabulary speech recognition”
 
 In arXiv preprint arXiv:1804.03209 , 2018
 

 
 [192] 
 Steven Weinberger and Stephen Kunath
 
 “The Speech Accent Archive: towards a typology of English accents”
 
 In Corpus-based studies in language use, language learning, and language documentation 
 
 Brill, 2011, pp. 265–281
 

 
 [193] 
 Bo Xiong, Haoqi Fan, Kristen Grauman and Christoph Feichtenhofer
 
 “Multiview pseudo-labeling for semi-supervised learning from video”
 
 In Proceedings of the IEEE/CVF International Conference on Computer Vision , 2021, pp. 7209–7219
 

 
 [194] 
 Han Xu et al.
 
 “To be robust or to be fair: Towards fairness in adversarial training”
 
 In International Conference on Machine Learning , 2021, pp. 11492–11501
 
 PMLR
 

 
 [195] 
 Hiromu Yakura and Jun Sakuma
 
 “Robust audio adversarial example for a physical attack”
 
 In arXiv preprint arXiv:1810.11793 , 2018
 

 
 [196] 
 Chaofei Yang, Qing Wu, Hai Li and Yiran Chen
 
 “Generative poisoning attack method against neural networks”
 
 In arXiv preprint arXiv: 1703.01340 , 2017
 

 
 [197] 
 Tien-Ju Yang, Dhruv Guliani, Françoise Beaufays and Giovanni Motta
 
 “Partial variable training for efficient on-device federated learning”
 
 In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2022, pp. 4348–4352
 
 IEEE
 

 
 [198] 
 Zhuolin Yang, Bo Li, Pin-Yu Chen and Dawn Song
 
 “Characterizing audio adversarial examples using temporal dependency”
 
 In arXiv preprint arXiv:1809.10875 , 2018
 

 
 [199] 
 Joanna Yau et al.
 
 “TILES-2019: A longitudinal physiologic and behavioral data set of medical residents in an intensive care unit”
 
 In Scientific Data 9.1 
 
 Nature Publishing Group UK London, 2022, pp. 536
 

 
 [200] 
 Wentao Yu et al.
 
 “Federated learning in ASR: Not as easy as you think”
 
 In Speech Communication; 14th ITG Conference , 2021, pp. 1–5
 
 VDE
 

 
 [201] 
 AmirAli Zadeh et al.
 
 “Multimodal language analysis in the wild: Cmu-mosei dataset and interpretable dynamic fusion graph”
 
 In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , 2018, pp. 2236–2246
 

 
 [202] 
 Valentina Zantedeschi, Maria-Irina Nicolae and Ambrish Rawat
 
 “Efficient defenses against adversarial attacks”
 
 In Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security , 2017, pp. 39–49
 

 
 [203] 
 Hongyi Zhang, Moustapha Cisse, Yann Dauphin and David Lopez-Paz
 
 “mixup: Beyond empirical risk minimization”
 
 In arXiv preprint arXiv: 1710.09412 , 2017
 

 
 [204] 
 Tuo Zhang et al.
 
 “FedAudio: A Federated Learning Benchmark for Audio Tasks”
 
 In arXiv preprint arXiv:2210.15707 , 2022
 

 
 [205] 
 Peijia Zheng, Zhiwei Cai, Huicong Zeng and Jiwu Huang
 
 “Keyword Spotting in the Homomorphic Encrypted Domain Using Deep Complex-Valued CNN”
 
 In Proceedings of the 30th ACM International Conference on Multimedia , 2022, pp. 1474–1483
 

 
 [206] 
 Kun Zhou, Berrak Sisman, Rui Liu and Haizhou Li
 
 “Seen and unseen emotional style transfer for voice conversion with a new emotional speech dataset”
 
 In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2021, pp. 920–924
 
 IEEE
 

 
 [207] 
 Han Zhu et al.
 
 “Decoupled Federated Learning for ASR with Non-IID Data”
 
 In Proc. Interspeech 2022 , 2022, pp. 2628–2632
 
 DOI: 10.21437/Interspeech.2022-720