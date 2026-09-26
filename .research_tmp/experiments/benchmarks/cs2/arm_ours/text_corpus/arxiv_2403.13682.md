Threats, Attacks, and Defenses in Machine Unlearning: A Survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2403.13682v5 [cs.CR] 17 Feb 2025 
 
 

# Threats, Attacks, and Defenses in Machine Unlearning: A Survey

 PubID:  pubid: 0000–0000/00$00.00 © 2021 IEEE 
 
 
 Ziyao Liu
 
    
 Huanyi Ye
 
    
 Chen Chen
 
    
 Yongsen Zheng
 
    
 Kwok-Yan Lam
 † † thanks: Ziyao Liu, Chen Chen, Huanyi Ye, Yongsen Zheng and Kwok-Yan Lam are with Nanyang Technological University, Singapore. E-mail: liuziyao@ntu.edu.sg, huanyi001@e.ntu.edu.sg, chen.chen@ntu.edu.sg, yongsen.zheng@ntu.edu.sg, kwokyan.lam@ntu.edu.sg. 

 Abstract 
 
 Machine Unlearning (MU) has recently gained considerable attention due to its potential to achieve Safe AI by removing the influence of specific data from trained Machine Learning (ML) models. This process, known as knowledge removal, addresses AI governance concerns of training data such as quality, sensitivity, copyright restrictions, and obsolescence. This capability is also crucial for ensuring compliance with privacy regulations such as the Right To Be Forgotten (RTBF). Furthermore, effective knowledge removal mitigates the risk of harmful outcomes, safeguarding against biases, misinformation, and unauthorized data exploitation, thereby enhancing the safe and responsible use of AI systems. Efforts have been made to design efficient unlearning approaches, with MU services being examined for integration with existing machine learning as a service (MLaaS), allowing users to submit requests to remove specific data from the training corpus. However, recent research highlights vulnerabilities in machine unlearning systems, such as information leakage and malicious unlearning, that can lead to significant security and privacy concerns. Moreover, extensive research indicates that unlearning methods and prevalent attacks fulfill diverse roles within MU systems. This underscores the intricate relationship and complex interplay among these mechanisms in maintaining system functionality and safety. This survey aims to fill the gap between the extensive number of studies on threats, attacks, and defenses in machine unlearning and the absence of a comprehensive review that categorizes their taxonomy, methods, and solutions, thus offering valuable insights for future research directions and practical implementations.

 
 
 
 Index Terms:  Machine unlearning, MLaaS, threats, attacks, defenses

 
 

## I Introduction 

 
 In recent years, the concept of knowledge removal in AI systems has garnered significant attention from both academia and industry, primarily due to security, privacy, and safety concerns. There are some examples as follows:

 
 • 
 
 Some training data may be considered sensitive, necessitating its removal from the model to prevent the leakage of sensitive information.

 

 • 
 
 In fields like healthcare, navigation, or forensics, outdated or incorrect training data must be removed from the model to maintain its safety and reliability.

 

 • 
 
 Training data protected by copyright requires the removal of knowledge gained from them from the trained machine learning (ML) model. This step is crucial to comply with commercial regulations and avoid conflicts before the model’s deployment or release.

 

 • 
 
 Additionally, data privacy regulations such as GDPR [ 1 ] , APPI [ 2 ] , and CCPA [ 3 ] emphasize data rights, including the Right To Be Forgotten (RTBF), allowing individuals to request the deletion of their personal data. This underscores the importance of effective knowledge removal to comply with these regulations.

 

 
 
 
 Fig. 1 : An illustration of machine unlearning. 
 
 
 In AI systems, removing knowledge requires the deletion of specific data and its effects from a trained ML model. As such, machine unlearning (MU) [ 4 ] emerged as a crucial enabler of this process. As depicted in Figure 1 , an initial, yet naive, approach to implementing MU is through retraining. This method involves discarding the current model and retraining it from scratch with the remaining data after the removal of the data to be unlearned. Nevertheless, this approach can incur substantial computational expenses, making it impractical for scenarios involving large-scale models or extensive datasets. In response to this challenge, a range of efficient MU methodologies have recently been developed, allowing the unlearned model to be obtained from the existing trained model without the need for retraining from scratch [ 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 ] .

 
 
 Furthermore, with the widespread adoption of machine learning as a service (MLaaS), AI service providers are working to integrate machine unlearning functionality into their offerings to meet the growing emphasis on the RTBF in privacy regulations and the increasing need for knowledge removal from AI models. However, research indicates that vulnerabilities are present in MLaaS systems that provide unlearning services. As illustrated in Figure 2 , adversarial users may launch various attacks at different stages of the MLaaS process with unlearning functionality, leading to significant security issues and privacy concerns (see detailed discussion in subsequent sections). These threats pose significant challenges to the safe deployment of machine unlearning systems.

 
 
 Fig. 2 : The overall workflow for machine unlearning systems. 
 
 
 The increasing emphasis on security and privacy in unlearning systems has led to a clearer recognition of their vulnerabilities and the threats they face, as revealed by related studies. Moreover, the complex roles of unlearning methods and the prevalence of attacks have become more evident, underscoring their intricate interaction in maintaining the functionality and safety of MU systems. Given this backdrop, a thorough examination of existing research on threats, attacks, and defenses in MU systems is indispensable. Such a comprehensive review would not only categorize their taxonomy, methods, and solutions but also provide valuable insights for future research directions and practical implementations. Thus, this survey is developed to address this existing gap, aiming to offer a holistic overview and critical analysis of the field.

 
 
 Comparison with other surveys. 
Recently, several studies begin to study and analyze machine unlearning methodologies, resulting in a clear categorization of the existing survey research. These categories include general surveys on machine unlearning, highlighted by works such as [ 11 , 4 , 12 , 19 , 22 , 13 , 23 , 24 ] , which cover a wide range of topics from basic concepts and methods to the challenges faced in the field. Additionally, specific attention has been given to federated unlearning (FU) through studies like [ 25 , 26 , 27 , 28 ] . These surveys focus on the unique issues and uses of federated unlearning compared to traditional unlearning approaches. Furthermore, potential risks and threats within federated unlearning are examined in [ 29 ] . Benchmarks and surveys on MU approaches over large language models (LLMs) are given in [ 30 , 31 , 32 , 33 , 34 ] .

 
 
 
 • 
 
 General surveys on MU: [ 11 , 4 , 12 , 19 , 13 , 22 , 23 , 24 ] 

 

 • 
 
 Surveys on FU: [ 25 , 26 , 27 , 28 , 29 ] 

 

 • 
 
 Benchmarks and surveys on MU for LLMs: [ 30 , 31 , 32 , 34 , 33 , 35 ] 

 

 
 
 
 To the best of our knowledge, no comprehensive survey currently tackles security and privacy issues within the context of machine unlearning, particularly focusing on threats, attacks, and defensive strategies. This gap in the literature strongly motivates our work in compiling this survey. Our focus on categorizing and scrutinizing these elements aims to illuminate strategies for protecting machine unlearning systems, thereby enhancing their robustness and safety.

 
 
 Summary of contributions. The main contributions of this survey are listed as follows:

 
 1. 
 
 We present a unified workflow for machine unlearning, and drawing from this framework as well as threat models, we propose a novel taxonomy of threats, attacks, and defenses within MU systems.

 

 2. 
 
 We carry out a detailed analysis of the current threats, attacks, and defenses in machine unlearning, along with an examination of the intricate relationships among unlearning methods, attacks, and defensive strategies.

 

 3. 
 
 We provide an in-depth discussion concerning potential vulnerabilities and threats in unlearning systems and the challenges in defenses against these threats and attacks, highlighting key areas for future research exploration.

 

 
 
 
 Organisation of the paper. The remainder of this paper is structured as follows: Section II outlines the key concepts fundamental to this survey. Section III introduces a unified MU workflow to support the analysis and taxonomy presented in the subsequent sections. Section IV provides an overview of the threats in MU systems and the various roles of attacks and defenses. Section V provides the threat models for machine unlearning systems. Building on this foundation, we conduct a thorough review of threats within the MU workflow, complete with a taxonomy, in Section VI . Section VII discusses the role of unlearning in defense mechanisms, while Section VIII examines how attacks can serve as a means for unlearning evaluation, followed by a summary in Section IX . Section X delves into challenges and potential directions for future research. Finally, Section XI provides a summary and concludes the paper.

 
 
 

## II Preliminaries 

 
 In this section, we provide the foundational concepts for understanding the topics covered in this survey. We introduce machine unlearning, followed by an overview of machine learning as a service. We also discuss key attacks on AI models, including poisoning attacks and membership inference attacks, to lay the groundwork for the threat and defense analysis in later sections.

 
 

### II-A Machine Unlearning 

 
 In machine unlearning, as illustrated in Figure 1 , the training dataset D D comprises two subsets, namely D u D_{u} and D r D_{r} , representing the data points to be unlearned and the remaining data points, respectively, wherein D r = D \ D u D_{r}=D\backslash D_{u} . Denote ℳ t = ℳ ⁡ ( D ) \mathcal{M}_{t}=\mathcal{M}(D) as the model trained on the entire dataset D D . The goal of MU is to produce an unlearned model ℳ u \mathcal{M}_{u} that closely resembles the model ℳ u ∗ \mathcal{M}_{u}^{*} trained solely on D r D_{r} , i.e., ℳ u ∗ = ℳ ⁡ ( D r ) \mathcal{M}_{u}^{*}=\mathcal{M}(D_{r}) .

 
 
 Machine unlearning can be classified as either exact unlearning or approximate unlearning [ 4 ] . Exact unlearning techniques typically involve retraining but limit the scope of data involved in order to enhance efficiency compared to naive retraining approaches. On the other hand, approximate unlearning adjusts the parameters of the existing trained model to create an unlearned model that approximates one that would be obtained through retraining from scratch. This approach often requires tradeoffs between unlearning performance and efficiency. Interested readers are referred to the relevant survey papers on machine unlearning, such as [ 4 , 12 , 13 , 25 , 28 ] , for more comprehensive details about MU methodologies and taxonomy.

 
 
 

### II-B Machine Learning as a Service 

 
 Machine Learning as a Service (MLaaS) is a cloud-based service model that provides machine learning tools and infrastructure as part of a broader service offering. In an MLaaS framework, service providers host machine learning models and processing power, allowing users to submit data and receive predictions or insights without needing to develop, train, or maintain their own models [ 36 ] . As a result, MLaaS has gained widespread adoption due to its ease of use, scalability, and accessibility, enabling organizations to focus on data and application logic rather than model development and infrastructure management. Current research efforts on MLaaS include but are not limited to privacy-preserving MLaaS [ 37 , 38 ] that ensure user data privacy during model training and prediction, scalability and efficiency to handle large-scale data and complex models with improved computational performance and resource management [ 39 , 40 ] , security and adversarial robustness to secure MLaaS systems against adversarial attacks like data poisoning and model evasion, enhancing system robustness. [ 41 , 42 ] .

 
 
 

### II-C Attacks on AI Models 

 
 Attacks on AI models refer to intentional actions taken by adversaries to manipulate, exploit, or undermine the performance and integrity of machine learning systems. These attacks can occur at various stages of the AI lifecycle, including data collection, training, and deployment. Common types of attacks include adversarial attacks [ 43 ] , where small, carefully crafted changes are made to input data to deceive the model into making incorrect predictions, and data poisoning attacks [ 44 ] , where malicious data is introduced into the training set to degrade the model’s performance. Other threats, such as model inversion [ 45 ] and membership inference [ 46 ] , aim to extract sensitive information about the training data or individuals whose data was used to train the model. Since data poisoning attacks and membership inference attacks are widely adopted in the related works reviewed in this survey, we dedicate additional space to the preliminaries of these two types of attacks to facilitate a better understanding of their mechanisms in attacks on MU systems and the defenses discussed in the subsequent sections.

 
 
 Poisoning attack. Poisoning attacks are a form of attack that aims to corrupt a machine learning model by injecting malicious data into its training set [ 44 ] . These attacks are broadly categorized into untargeted and targeted poisoning attacks. Untargeted poisoning attacks aim to degrade the overall performance of the model by introducing enough corrupted data to lower its accuracy across a wide range of inputs. In contrast, in targeted poisoning attacks, the adversary focuses on manipulating the model to misclassify specific inputs or behave incorrectly in particular situations, such as causing a facial recognition system to misidentify a single person. A more specific type of targeted attack, known as a backdoor attack (BA) [ 47 ] , embeds a distinct pattern or ’trigger’ into portions of the training data [ 47 ] . This trigger can be a small patch or sticker visible to humans [ 48 ] , or a subtle perturbation in benign samples that is indistinguishable from human inspection [ 49 ] . Once the model is trained or fine-tuned on this compromised data, it behaves normally with standard inputs. However, when it encounters an input containing the covert trigger, the model exhibits malicious behavior aligned with the attacker’s intentions. Note that poisoning attacks and backdoor attacks can be particularly dangerous in MU systems during the unlearning process, as they can be more stealthy and difficult to detect. A detailed review of related works and further discussion on these threats will be provided in Section VI-B .

 
 
 Membership inference attack. Membership inference attack (MIA) allows an adversary to obtain an attack model to determine whether a specific data point was part of the model’s training set [ 46 ] . Given access to the model’s predictions, attackers can exploit patterns or overfitting to infer if a particular individual’s data was used in training, which can lead to privacy violations. Remarkably, MIA does not require knowledge of the target model’s specific architecture or the distribution of its training data. Relying on the shadow models, a series of shadow training datasets D 1 ′ , ⋯ , D k ′ D^{\prime}_{1},\cdots,D^{\prime}_{k} and disjointed shadow test datasets T 1 ′ , ⋯ , T k ′ T^{\prime}_{1},\cdots,T^{\prime}_{k} can be synthesized to mimic the behavior of the target model so as to train the attack model. MIA can be used both to explore information leakage by analyzing the differences between the trained and unlearned models and to evaluate whether the target data has been successfully unlearned. This will be discussed in detail in Section VI and Section VIII .

 
 
 
 

## III Workflow of Machine Unlearning Systems 

 
 In this section, we will define a unified workflow for MLaaS with unlearning services, which will aid in understanding the threats and attacks within these systems, as well as the corresponding defensive strategies.

 
 
 As illustrated in Figure 2 , the structure of machine unlearning systems designed to provide unlearning services typically includes a server and various participants. In specific, the server plays key roles in the system, including:

 
 • 
 
 Model developer: responsible for conducting model training based on the training data.

 

 • 
 
 Service provider: responsible for performing inference based on provided inputs.

 

 
 Note that these roles can be allocated to different entities depending on the application scenario [ 50 ] . In addition, the roles of participants within a MU system are classified as follows:

 
 • 
 
 Data contributors: responsible for providing data to construct the training dataset.

 

 • 
 
 Requesting users: possess the capability to submit unlearning requests.

 

 • 
 
 Accessible users: utilize the service for inferences.

 

 
 Note that akin to the server’s multi-functional role, participants within unlearning systems can embody various roles. For instance, it is common for data contributors to also have the authority to submit unlearning requests, in compliance with RTBF regulations. Additionally, the workflow of machine unlearning systems can be structured into three phases, which we categorize as follows:

 
 • 
 
 Training phase: the ML model is trained using the data supplied by data contributors.

 

 • 
 
 Unlearning phase: the server conducts unlearning algorithms to remove the effects of certain data points from the model’s knowledge base, in response to unlearning requests from requesting users.

 

 • 
 
 Post-unlearning phase: the server provides inference services based on unlearned models to accessible users.

 

 
 
 
 In the context of machine-learning-as-a-service (MLaaS), the server may receive requests for unlearning and inference simultaneously. This indicates a possible overlap between the unlearning phase and the post-unlearning phase, where the procedures for managing different types of requests become more complex [ 51 ] .

 
 
 

## IV Overview of Threats, Attacks, and Defenses in MU Systems 

 
 As mentioned earlier, vulnerabilities are present in machine unlearning systems, where adversaries can launch various attacks at different phases, posing significant threats to the safe deployment of unlearning services within MLaaS. Besides threats and attacks on machine unlearning systems, many studies show that unlearning and attacks serve dual roles. For instance, unlearning can act as a mechanism to recover models from backdoor attacks, while backdoor attacks themselves can serve as an evaluation metric for unlearning effectiveness. This reveals a complex interaction between these elements, which are essential for system safety and functionality. In this section, we offer an overview of threats, attacks, and defenses in machine unlearning systems. We explore these topics in depth in Sections VI , VII , and VIII , where we present a detailed taxonomy as shown in Figure 3 .

 
 
 Fig. 3 : A taxonomy of threats, attacks, and defenses in machine unlearning.
 
 
 
 Threats. As illustrated in Figure 2 , vulnerabilities are present across all phases of the unlearning system. In the training phase, for example, data contributors can maliciously craft the training data in a way that prepares for future attacks. In the unlearning phase, requesting users have the opportunity to submit malicious unlearning requests that could be designed to achieve their goals after the unlearning process. Moreover, during the post-unlearning phase, participants with access to both the trained and the unlearned models might exploit this dual access to uncover additional information leakages, thus exposing further vulnerabilities created by the unlearning process. A more detailed discussion on the various threats that may arise is available in Section VI .

 
 
 Attacks. The threats outlined previously can be exploited to execute attacks. For instance, a data contributor might poison the training dataset, and bypass the detection by incorporating mitigation data during the training phase, which could then be removed via unlearning in the unlearning phase [ 52 ] . A requesting user could craft an unlearning request to unlearn more information than expected, thereby manipulating the system to their advantage [ 50 ] . Furthermore, an accessible user could perform enhanced membership inference attacks [ 46 , 53 , 54 ] by leveraging their access to both the trained and unlearned models, exploiting the differential insights provided by comparing the two models [ 55 ] . These attacks, along with threats to unlearning systems, are discussed in detail in Section VI .

 
 
 Beyond the attacks on machine unlearning systems, extensive research has demonstrated that certain prevalent attack strategies, like backdoor attacks [ 56 ] and membership inference attacks [ 46 ] , can be adapted for unlearning evaluations, such as privacy leakage audit [ 55 ] , model robustness assessment [ 50 ] , and proof of unlearning [ 57 ] . The analysis of attacks used for evaluating unlearning is provided in Section VII .

 
 
 Defenses. Potential defense methods against attacks on MU systems are explored in conjunction with attack strategies. For example, performing membership checks on data targeted for unlearning can ensure that the data specified in unlearning requests genuinely exists in the training dataset [ 50 ] . Implementing differential privacy [ 58 , 59 ] can help prevent membership inference attacks, with a sacrifice of model performance [ 55 ] . Moreover, anomaly detection through model monitoring in the unlearning phase can effectively identify potential issues [ 60 ] . A corresponding discussion, along with an analysis of threats and attacks, is given in Section VI .

 
 
 Additionally, unlearning can serve as a method of defense. It is capable of removing backdoors from trained machine learning models [ 61 , 62 , 63 ] , and can be employed to achieve value alignment, thereby providing a defense mechanism against potential AI safety issues [ 64 , 34 , 65 ] . The dual role of unlearning as a defense mechanism is analyzed in Section VII .

 
 
 In summary, threats, attacks, and defenses within machine unlearning systems exhibit a complex relationship, revealing an intricate interplay among these elements in upholding system functionality and safety.

 
 
 

## V Threat Models 

 
 In this section, we define the threat models based on our proposed unified MU workflow described in Figure 2 , regarding the role of attackers in unlearning services, the attackers’ goals, their knowledge of the MU system, and the phases during which attacks occur. This analysis of threat models will provide a deeper understanding of the existing related works discussed in the subsequent sections.

 
 

### V-A Attack Roles 

 
 Threats and attacks within the machine unlearning workflow can arise from participants adopting various adversarial roles. These roles can be categorized into three distinct classes, as presented in Section I , outlined below:

 
 
 
 • 
 
 Data contributors (R1) : Individuals responsible for providing data to construct the training dataset.

 

 • 
 
 Requesting users (R2) : Individuals who have the ability to submit unlearning requests.

 

 • 
 
 Accessible users (R3) : Individuals with access to the model service, enabling them to utilize the model for making inferences.

 

 
 
 
 Note that attackers within unlearning systems can fulfill various attack roles. A requesting user, for example, usually has access to the model service for inference purposes. Similarly, requesting users can also serve as data contributors. Generally, the more roles an attacker holds, the stronger their potential for launching effective attacks becomes.

 
 
 

### V-B Attack Goals 

 
 Attack goals within unlearning systems also vary and can generally be categorized as follows:

 
 
 
 • 
 
 Untargeted (G1) : Intend to mislead the unlearned model into generating incorrect predictions without targeting a specific outcome.

 

 • 
 
 Targeted (G2) : Aim at inducing the unlearned model to produce incorrect predictions or behave in a specific, predetermined way.

 

 • 
 
 Privacy leakage (G3) : Seeks to extract additional information about the data requested to be unlearned.

 

 • 
 
 Others (G4) : Goals may include increasing the unlearning process’s computational cost, impacting the unlearned model’s fairness and utility, etc.

 

 
 
 
 

### V-C Adversarial Knowledge 

 
 Additionally, attackers may possess varying levels of adversarial knowledge for an attack, which can be classified into three categories. Generally, the more knowledge an attacker has, the more potent the attack becomes.

 
 
 
 • 
 
 White-box (K1) : The attacker possesses comprehensive knowledge of the model’s architecture, parameters, training and unlearning algorithms, and data.

 

 • 
 
 Grey-box (K2) : The attacker has access to some elements such as the model’s architecture, parameters, training and unlearning algorithms, or data, but not to all of them.

 

 • 
 
 Black-box (K3) : The attacker has no knowledge of the model, including its architecture and parameters, and cannot access or modify the model’s training process.

 

 
 
 
 

### V-D Attack Phases 

 
 Attacks and threats can occur at different stages within an ML system with unlearning services, and can be categorized into the following three phases:

 
 
 
 • 
 
 Training phase (P1) : Attackers, serving as data contributors, may manipulate or craft the training dataset to prepare for an attack during the later phases.

 

 • 
 
 Unlearning phase (P2) : Attackers may submit malicious unlearning requests to initiate attacks, which may or may not be based on preparations in the training phase.

 

 • 
 
 Post-unlearning phase (P3) : Attackers may launch attacks leveraging the unlearned model and information acquired during the previous phases.

 

 
 
 
 
 

## VI Threats in Unlearning 

 
 In this section, we explore the threats inherent to machine unlearning systems. Our focus encompasses the threats of information leakage through machine unlearning, the impact of malicious unlearning practices, and other vulnerabilities that may arise within unlearning systems. We offer a thorough analysis of their methodologies based on their respective threat models (see Section V for more details) and discuss potential defense mechanisms to mitigate these risks.

 
 

### VI-A Information Leakage 

 
 As previously noted, machine unlearning can introduce specific forms of information leakage. This leakage often stems from (i) discrepancies between the trained and unlearned models, or from (ii) the insights derived from the dependency between the model and external knowledge. In this section, we will explore these two sources of information leakage. It is important to emphasize that our discussion is centered on information leakage uniquely associated with machine unlearning, rather than the general leakage issues found in standard ML systems.

 
 
 Leakage from model discrepancy. Model discrepancy refers to the differences observed between the trained model and the unlearned model. Attackers can take advantage of this variance to extract additional information about the unlearned data. This could involve determining the membership of specific unlearned data points [ 55 , 66 , 51 , 67 ] or attempting to reconstruct the data that was intended to be unlearned [ 67 ] .

 
 
 In particular, [ 55 ] utilizes a membership inference attack strategy to determine whether a given input was part of the unlearned data. For a specific input, both the trained and unlearned models produce confidence values represented as vectors. By strategically combining these vectors of confidence values, an independent attack model can be trained to deduce the presence of the input within the unlearned data, essentially inferring its membership. Compared to [ 55 ] , the membership inference attack outlined in [ 66 ] necessitates less information, relying solely on the Top-1 value of the confidence vectors, namely, the inferred label. To accomplish this, the attacker strategically perturbs the input to observe the changes in the outputs of both the trained and unlearned models, thereby deducing the membership of the input. In [ 67 ] , a general framework for inference and reconstruction attacks is formally established, discussing various instances across different machine learning tasks. Data construction attacks are also investigated in [ 10 ] , where the model discrepancy is used as estimated gradients. Based on this estimation, attacks through deep leakage from gradients [ 68 , 69 ] can be employed to achieve data reconstruction. Additionally, [ 10 ] demonstrates that label inference attacks can be effective even with only access to prediction services, without needing the model parameters, in a black-box setting. The information leakage from model discrepancy has been shown to be effective for textual tasks in [ 70 ] . Based on the proposed attack methods, attackers can infer the membership of unlearned data in a black-box setting and reconstruct the unlearned data with white-box access to the two versions of the model.

 
 
 Note that the privacy leakages mentioned above occur in the post-unlearning phase when accessible users can interact with the model service for inferences. However, [ 51 ] highlights a new privacy risk from the Rignt-To-Be-Forgetten perspective in an MLaaS setting where the unlearning phase for one request and the post-unlearning phase for another inference request may overlap, i.e., inference requests and unlearning requests may arrive at the server within a very close timeframe. In such scenarios, prioritizing the “unlearning-request-first” could lead to service obsolescence, whereas handling the “inference-request-first” enables an attacker to gain knowledge about a specific input, which can be exploited by attacks like membership inference attacks. To counter this threat, strategies have been developed to evaluate the urgency of processing unlearning requests immediately. This evaluation is based on whether the unlearning disrupts the consistency between the trained and unlearned models.

 
 
 Leakage from knowledge dependency. Knowledge dependency stems from the natural relationships between the model and external knowledge sources, such as the association between successive unlearning requests and the unlearned model [ 71 ] , as well as the correlation between open-source model parameters and the embedding of prompts in LLMs [ 72 ] . Leveraging these dependencies to gain extra knowledge, attackers may initiate various attacks leading to information leakage.

 
 
 In [ 71 ] , two types of knowledge dependency are described within machine unlearning systems.
The first one involves adaptive requests [ 73 ] , indicating that an unlearning request to remove certain data points may influence the removal of other data points. This relationship can be exploited through a series of successive unlearning requests, gathering sufficient knowledge to ascertain the membership of specific data. The second instance of knowledge dependency arises from unlearning algorithms caching partial computations to expedite processing. This method can unintentionally reveal information about data meant to be deleted across multiple releases. The dependency between the parameters of unlearned large models and the embedding of prompts is investigated in [ 72 ] . By introducing adversarial perturbations to the embeddings of prompt tokens, it is possible to instruct the large model to reveal knowledge that was supposed to be erased through unlearning.

 
 
 Defenses. 
To mitigate privacy attacks and subsequent information leakage, the essential strategy is to limit the exploitable knowledge for attackers. For model discrepancy-related leakage, reducing the amount of information provided by the models, such as only showing top-k or top-1 confidence scores [ 55 ] , applying techniques like label smoothing and differential privacy [ 55 , 66 , 67 ] , or strategic postponed unlearning [ 51 ] , are effective approaches. Regarding leakage from knowledge dependency, solutions vary. Specifically, prevent unlearning algorithms from caching can address leaks from cached data, and redefining privacy metrics for adaptive requests can help measure and control privacy risks under those scenarios.

 
 
 
 
 
 (a) Direct unlearning attacks. 
 
 
 (b) Preconditioned unlearning attacks. 
 
 Fig. 4 : 
An illustration of direct and preconditioned unlearning attacks via decision boundary manipulation. (a) Direct unlearning attacks occur solely in the unlearning phase, with no manipulation of training data. (b) Preconditioned unlearning attacks involve strategic manipulation of training data. 
 
 
 

### VI-B Malicious Unlearning 

 
 As noted earlier, attackers can engage in malicious unlearning by submitting crafted unlearning requests during the unlearning phase. It is important to note that such an attack happens in the unlearning phase, but the attacker may or may not need some preparation, e.g., manipulating the training data, in the training phase. We call malicious unlearning attacks that do not need preparation “direct unlearning attacks” and those that require preparation “preconditioned unlearning attacks”. Next, we will explore the methods and principles of these two types of attacks.

 
 
 Direct unlearning attacks. Direct unlearning attacks occur solely in the unlearning phase, without requiring the manipulation of training data during the training phase, while attackers may or may not possess knowledge of the training data. These attacks are designed to be either untargeted, aiming to degrade the overall performance of the model after unlearning [ 50 ] , or targeted, with the intention of causing the unlearned model to misclassify target inputs with predefined features [ 74 ] .

 
 
 Specifically, [ 50 ] demonstrates how untargeted attacks can be conducted through over-unlearning techniques in a black-box setting. This is based on the critical observation that models struggle to predict data points close to the decision boundary, as even minor changes in the input could result in different predictions. As depicted in Figure 4(a) , after a malicious unlearning process, a number of data points end up closer to the decision boundary than in normal unlearning scenarios. Consequently, these data points are more prone to misclassification compared to the normal case. Hence, the unlearned model obtained after malicious unlearning exhibits lower classification accuracy than a model that has undergone normal unlearning. For crafting such malicious unlearning requests, attackers leverage the model’s inference service to acquire knowledge about the trained model. This knowledge enables them to formulate malicious unlearning requests employing methods of adversarial perturbation [ 75 , 76 ] .

 
 
 Distinct from the untargeted attacks described in [ 50 ] , targeted attacks can be executed with additional knowledge of the training data in a grey-box or white-box setting, as detailed in [ 74 ] . Unlike the approach in [ 50 ] that shifts the decision boundary closer to remaining data points, targeted attacks manipulate the decision boundary to selectively cross through remaining data points, classifying specific data points with predefined features into a target class. As outlined in [ 74 ] , to generate these malicious unlearning requests, attackers exploit the model’s inference service along with training data to which they have access. They address an optimization problem to identify the optimal perturbation and the data points to be perturbed. These selected data points, once perturbed, are then submitted as part of the malicious unlearning requests.

 
 
 Preconditioned unlearning attacks. Unlike direct unlearning attacks, preconditioned unlearning attacks take a more strategic approach by manipulating training data during the training phase. These attacks are typically designed to carry out complex and stealthy targeted attacks.

 
 
 As depicted in Figure 4(b) , a typical preconditioned unlearning attack is executed through the following steps:

 
 
 
 1. 
 
 The attackers incorporate poisoned data D p D_{p} and mitigation data D m D_{m} into a clean dataset D c D_{c} , to ensemble a training dataset D = D p ∪ D m ∪ D c D=D_{p}\cup D_{m}\cup D_{c} .

 

 2. 
 
 The server trains the ML model based on D D and returns the model ℳ \mathcal{M} .

 

 3. 
 
 The attackers submit requests to unlearn the mitigation data D m D_{m} .

 

 4. 
 
 Upon receiving the requests, the server carries out the unlearning process and returns the unlearned model ℳ ^ \hat{\mathcal{M}} , which effectively corresponds to a model trained on the dataset D p ∪ D c D_{p}\cup D_{c} .

 

 
 
 
 Thus, after the unlearning process removes the mitigation data, the model is left vulnerable to the poisoned data inserted during the training phase. This exposes the unlearned model to the influence of the initial poisoning, culminating in a successful preconditioned unlearning attack.

 
 
 A number of studies have successfully executed preconditioned unlearning attacks, including [ 60 , 52 , 77 ] . The primary distinction between [ 52 ] and [ 60 ] lies in their approaches to generating poisoned and mitigation data during the training phase. In [ 52 ] , poison generation is approached as a bilevel optimization problem, as described in [ 78 ] , with mitigation data produced either through label flipping or gradient matching to neutralize the poison’s effect. On the other hand, [ 60 ] outlines two methodologies for crafting the training dataset: (i) generating poisoned data by perturbing sampled clean data, with mitigation data created using the same sampled perturbed clean data but with ground-truth labels, akin to the label flipping technique in [ 78 ] , and (ii) a different strategy for poisoned data generation inspired by Badnets [ 79 ] , where the mitigation data contains a trigger that exerts a stronger influence on the model, hence concealing the effects of the poisoned data. Camouflaged sample generation based on data influence is explored in [ 80 ] , along with a comprehensive performance evaluation of various machine unlearning methods.

 
 
 In addition to the typical preconditioned unlearning attacks detailed in [ 60 , 52 ] , another strategy is introduced in [ 77 ] where attackers deliberately perturb certain pieces of training data, which are subsequently used in model training. In the unlearning phase, attackers strategically submit requests for the removal of the perturbed data points. This approach takes advantage of the knowledge gained from the model’s inference service and the perturbed training data, thereby executing an effective form of preconditioned unlearning attack. Another type of preconditioned unlearning attack is proposed in [ 81 ] , where attackers deliberately generate training data. The overall model usability will degrade after unlearning these data.

 
 
 Defenses. Defensive strategies to counteract malicious unlearning have been explored in the studies previously mentioned. A notable observation is a significant distinction in defense methods when addressing direct versus preconditioned unlearning attacks. The distinction arises because, in direct unlearning attacks, the data submitted for unlearning are not part of the original training dataset. In contrast, in preconditioned unlearning attacks, the data requested to be unlearned are indeed from the training dataset. For instance, to prevent malicious unlearning requests in direct unlearning attacks, one can compare the Hash value of the unlearned data and training data or make the membership inference as pointed out in [ 50 , 81 ] .
In the case of preventing malicious unlearning requests during preconditioned unlearning attacks, membership checking is ineffective because the data being unlearned is part of the original training dataset. Consequently, [ 60 ] suggests detection strategies based on metrics that compare the model’s performance on clean data versus mitigation data. If the discrepancy exceeds a certain threshold, the unlearning request may be denied. On a different note, [ 77 ] proposes not directly rejecting unlearning requests but instead dropping malicious unlearned data from requests that exhibit a gradient dissimilarity compared to other instances within their class.

 
 
 

### VI-C Other Vulnerabilities 

 
 In addition to privacy leakages resulting from unlearning and threats induced by malicious unlearning practices, various other vulnerabilities within machine unlearning have been explored from multiple perspectives.

 
 
 In [ 82 ] , the concept of a “slow-down attack” is explored, identifying it as a type of poisoning attack aimed at decelerating the unlearning process. This attack is particularly relevant in the context of certified removal [ 83 ] , a system that decides between conducting a full retraining or an approximate update based on specific triggers. By strategically crafting poisoned data in the training data, one can minimize the interval or the number of unlearning requests processed through approximate updates before necessitating full retraining. Since a full retrain is considerably more time-consuming than an approximate update, such a “slow-down attack” significantly diminishes the overall efficiency of unlearning. The target of creating this poisoned data is formulated as a bilevel optimization problem [ 84 ] , where locally optimal solutions can be obtained using gradient-based methodologies.

 
 
 Other studies highlight that unlearning algorithms themselves might introduce side effects in specific domain applications. For example, [ 85 ] discusses the impact on fairness in LLMs, noting that ensemble unlearning methods, like those detailed in [ 86 ] , can affect fairness. A proposed mitigation approach involves adjusting the LLMs’ outputs using bias mitigation techniques outlined in [ 87 ] . Meanwhile, [ 88 ] , focusing on Graph Neural Networks (GNNs), reveals that unlearning algorithms, such as [ 89 ] , can lead to over-forgetting. This phenomenon occurs when the unlearning process inadvertently removes more information than necessary, significantly reducing the performance of the remaining edges. Since over-forgetting stems from the design of the unlearning algorithm itself, [ 88 ] suggests modifying the unlearning algorithms to prevent such outcomes.

 
 
 
 
 
 | 
 | 
 Roles | 
 Goals | 
 Knowledge | 
 Phases | 
 | 
 | 

 
 Threats | 
 Ref. | 
 
 
 R1 
 | 
 
 
 R2 
 | 
 
 
 R3 
 | 
 
 
 G1 
 | 
 
 
 G2 
 | 
 
 
 G3 
 | 
 
 
 G4 
 | 
 
 
 K1 
 | 
 
 
 K2 
 | 
 
 
 K3 
 | 
 
 
 P1 
 | 
 
 
 P2 
 | 
 
 
 P3 
 | 
 Techniques | 
 Defenses | 

 
 | 
 | 
 [ 55 ] | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 Membership inference attacks 
 | 
 
 
 
 
 
 Limiting output knowledge (S3); 
 
 Label smoothing (S3); 
 
 Differential privacy (S3) 
 
 | 

 
 | 
 | 
 [ 66 ] | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 Membership inference attacks 
 | 
 
 
 
 
 
 Label smoothing (S3); 
 
 Differential privacy (S3) 
 
 | 

 
 | 
 | 
 [ 67 ] | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 
 
 
 Deletion inference attacks; 
 
 Deletion reconstruction attacks 
 
 | 
 
 
 Differential privacy (S3) 
 | 

 
 | 
 | 
 [ 51 ] | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 Obsolescent inference services 
 | 
 
 
 Strategic postponed unlearning (S2,S3) 
 | 

 
 | 
 | 
 [ 10 ] | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 
 
 
 Unlearning inversion attacks 
 
 | 
 
 
 
 
 
 Parameter obfuscation (S1); 
 
 Model pruning (S3); 
 
 Fine-tuning (S3) 
 
 | 

 
 | 
 | 
 [ 90 ] | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 Reconstruction attacks 
 | 
 
 
 Differential privacy (S3) 
 | 

 
 | 
 
 
 
 Model 
 
 Discrepancy 
 | 
 [ 70 ] | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 Textual unlearning leakage attack 
 | 
 
 
 NA 
 | 

 
 | 
 | 
 [ 71 ] | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 
 
 
 Submitting adaptive requests; 
 
 Storing secret models 
 
 | 
 
 
 
 
 
 Regulated unlearning algorithms (S1); 
 
 Redefined certified removal (S1) 
 
 | 

 
 
 
 
 Information 
 
 Leakage 
 | 
 
 
 
 Knowledge 
 
 Dependency 
 | 
 [ 72 ] | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 Adversarial embeddings 
 | 
 
 
 NA 
 | 

 
 | 
 | 
 [ 50 ] | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 Over-unlearning 
 | 
 
 
 
 
 
 Membership checking (S1); 
 
 Model monitoring (S2) 
 
 | 

 
 | 
 | 
 [ 74 ] | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 Perturbing unlearning requests 
 | 
 
 
 NA 
 | 

 
 | 
 
 
 
 Direct 
 
 Unlearning 
 
 Attacks 
 | 
 [ 91 ] | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 Backdoor attacks 
 | 
 
 
 NA 
 | 

 
 | 
 | 
 [ 91 ] | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 Backdoor attacks 
 | 
 
 
 NA 
 | 

 
 | 
 | 
 [ 52 ] | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 Camouﬂaged attacks 
 | 
 
 
 NA 
 | 

 
 | 
 | 
 [ 60 ] | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 Camouﬂaged attacks 
 | 
 
 
 Model monitoring (S2) 
 | 

 
 | 
 | 
 [ 77 ] | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 Perturbing training data 
 | 
 
 
 Malicious requests detection (S1) 
 | 

 
 
 
 
 Malicious 
 
 Unlearning 
 | 
 
 
 
 Preconditioned 
 
 Unlearning 
 
 Attacks 
 | 
 [ 81 ] | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 Unlearning usability attack 
 | 
 
 
 Model monitoring (S2) 
 | 

 
 | 
 | 
 [ 80 ] | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 Camouﬂaged attacks 
 | 
 
 
 NA 
 | 

 
 | 
 [ 82 ] | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 Poisoning training data 
 | 
 
 
 NA 
 | 

 
 | 
 [ 85 ] | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 Ensemble unlearning 
 | 
 
 
 Output modification (S3) 
 | 

 
 Other Vulnerabilities | 
 [ 88 ] | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 
 
 ✓ 
 | 
 | 
 | 
 | 
 
 
 ✓ 
 | 
 | 
 
 
 Over-forgetting 
 | 
 
 
 Improved unlearning algorithms (S1) 
 | 

 
 TABLE I : Summary of threats in machine unlearning systems. Attack Roles : Data contributors (R1), Requesting users (R2), Accessible users (R3); Attack Goals : Untargeted (G1), Targeted (G2), Privacy leakage (G3), Others (G4); Adversarial Knowledge : White-box (K1), Grey-box (K2), Black-box (K3); Attack Phases : Training phase (P1), Unlearning phase (P2), Post-unlearning phase (P3); Defense Stages : Pre-unlearning stage (S1), In-unlearning stage (S2), Post-unlearning stage (S3). 
 
 
 

### VI-D Summary 

 
 So far, we have explored threats inherent to machine unlearning systems and presented a taxonomy that includes:

 
 • 
 
 Information leakage

 

 • 
 
 Malicious unlearning

 

 • 
 
 Other vulnerabilities

 

 
 
 
 Additionally, we have conducted an in-depth analysis of their methodologies, guided by the threat models. A summary of threats in machine unlearning is provided in Table I . Furthermore, in discussing defense strategies against these threats and attacks, a detailed review reveals that these strategies can be categorized based on the stage at which the defense is implemented. The classifications are as follows:

 
 
 
 • 
 
 Pre-unlearning: at this stage, defense methods usually focus on detecting malicious requests prior to initiating the unlearning process, or implementing regulations to govern how unlearning is conducted.

 

 • 
 
 In-unlearning: at this stage, defense strategies often focus on monitoring changes in the model to potentially halt the unlearning process if anomalies are detected.

 

 • 
 
 Post-unlearning: at this stage, defense methods here generally aim at protecting information leaked from unlearned models, or recovering the model to its state before the attack, for example, by adding differential perturbations to the unlearned models, rolling back the model, or removing backdoors for model recovery, leveraging unlearning techniques (see Section VII for more details).

 

 
 
 
 Note that defense strategies during the in-unlearning and post-unlearning stages are highly flexible, allowing for the use of diverse algorithms as drop-in solutions to mitigate both direct and preconditioned unlearning attacks. Conversely, the defensive approaches used in the pre-unlearning stage vary significantly, tailored to address specific threats and attacks.

 
 
 
 

## VII Defense through Unlearning 

 
 As previously discussed, beyond facilitating knowledge removal, unlearning can also act as a defensive mechanism, collectively enhancing AI safety. Therefore, we dedicate this section to a focused discussion on unlearning as a form of defense. Specifically, we will explore two primary targets of unlearning in this defensive context, including model recovery and value alignment.

 
 

### VII-A Model Recovery 

 
 Data poisoning attacks on machine learning models have been extensively studied over the years, revealing that both untargeted and targeted attacks can be effectively detected either during or after the training process [ 49 , 92 , 93 , 94 , 95 , 96 , 97 ] . However, these detection methods typically need substantial model updates to make confident decisions. As a result, by the time attacks are identified, the poisoned data may have already impacted the ML model [ 62 , 98 , 99 ] . This scenario underscores the necessity for recovering an accurate model from a poisoned one that has been compromised after the detection of such attacks.

 
 
 In this context, machine unlearning presents itself as an intuitive approach for model recovery, enabling the targeted removal of identified poisoned data. For example, [ 63 ] demonstrates backdoor removal by inverting the trigger to retrain infected ML models, a technique also utilized in [ 100 ] . A more complex strategy for eliminating backdoors is detailed in [ 61 ] , where the process begins with identifying the trigger pattern based on which then conducting unlearning via gradient ascent. [ 101 ] focuses on model recovery through unlearning, based solely on limited access to poisoned data. In [ 65 ] , sparsity-aware unlearning is executed by initially pruning the model, followed by the unlearning process. Diverging from these approaches, [ 102 ] and [ 103 ] explore the effects of unlearning on adversarial and backdoor attacks, each proposing different methods to balance trade-offs. The former focuses on unlearning universal adversarial perturbations, while the latter targets the unlearning of shared adversarial examples. The concept of model recovery through unlearning has also been explored in various other settings, including multimodal environments as seen in [ 104 ] , and federated learning scenarios, as discussed in [ 62 , 105 ] .

 
 
 

### VII-B Value Alignment 

 
 The concept of value alignment plays a pivotal role in enhancing AI safety, underscoring the importance of aligning machine learning models with human values and ethical standards. Given its significance, a wide array of research has been developed to leverage machine unlearning techniques as a strategic approach to achieve value alignment. This focus aims to mitigate risks and ensure that AI systems operate in a manner consistent with societal norms and individual preferences, highlighting unlearning as a key tool in the pursuit of safer AI integration.
 In these efforts, unlearning is applied in various innovative ways to align AI operations with ethical and legal standards. For instance, attribute unlearning [ 106 ] is used to remove sensitive attributes for privacy compliance. Similarly, unlearning with corresponding detection methods are applied to LLMs to remove behaviors and content that are deemed inappropriate or should be avoided, effectively eliminating illegal, poor quality, copyrighted, or harmful content [ 34 , 107 , 108 , 109 , 110 , 111 , 112 , 113 ] . In a separate vein, the work outlined by Isonuma et al. [ 114 ] introduces a distinctive application of unlearning principles aimed at identifying the training datasets that have the most profound influence on generating harmful content. Through a process of systematically unlearning each dataset via gradient ascent, the repercussions on the model’s propensity to generate harmful content can be meticulously evaluated post-unlearning.

 
 
 
 

## VIII Evaluating Unlearning through Attacks 

 
 Much like a coin has two sides, attacks on unlearning systems can be both a threat and an effective tool for evaluation. This dual role enables the auditing of privacy leakage, the assessment of model robustness, and the provision of proof of unlearning. In this section, we provide an overview of methods for evaluating unlearning through attacks, highlighting how these aspects of attacks can deepen our understanding and enhance the effectiveness of unlearning systems.

 
 

### VIII-A Audit of Privacy Leakage 

 
 As outlined in Section VI-A , machine unlearning systems are susceptible to threats of information leakage. Attackers can exploit discrepancies between the trained and unlearned models, or inherent vulnerabilities within the unlearning system, to initiate inference attacks [ 55 , 66 , 67 , 51 ] , or to reconstruct the unlearned data [ 71 , 72 ] .

 
 
 Therefore, these attacks can function as instruments to audit privacy leakage from the unlearning system after the unlearning process, representing common methodologies in privacy leakage assessment [ 115 , 116 ] . Metrics such as the Area Under the Curve (AUC) and Attack Success Rate (ASR) are widely adopted to quantify the effectiveness of privacy attacks, thereby indicating the level of privacy leakage [ 55 , 66 , 67 , 51 ] . Additionally, tracing the unlearning performance relative to the leakage of specific data with certain features can help measure overall privacy leakage [ 117 ] . Moreover, privacy degradation metrics, which compare against classical attacks to ML models, can quantify additional privacy leakage beyond what is typical with a single model [ 55 ] .

 
 
 

### VIII-B Assessment of Model Robustness 

 
 Similarly, malicious unlearning can assess the robustness of the unlearned model. The foundational principle of this assessment approach is akin to that used in adversarial attacks, as detailed in works such as [ 118 , 119 , 120 , 121 ] . For example, [ 50 ] describes a method of malicious unlearning via over-unlearning, where perturbed data are submitted in the unlearning request to alter the classifier’s decision boundary. This strategy of pushing the decision boundary allows for evaluating the impact of unlearned data points near the boundary on model performance, as well as the effect of data points that challenge the model’s prediction capabilities. Another example occurs in the context of LLMs [ 72 ] , where attackers insert adversarial embeddings into the prompts’ embeddings, aiming to extract supposedly unlearned knowledge from the model. Such attacks provide a means to gauge the model’s resilience against malicious input embeddings. On a different note, [ 122 ] and [ 123 ] treat data removal as an adversarial perturbation to the overall training dataset. In this case, the concept of randomized smoothing [ 124 ] can be adopted to measure the model robustness against unlearned data [ 122 ] or unlearned users in a federated setting [ 123 ] , hence establishing a certified budget for data removals but retains the model robustness.

 
 
 

### VIII-C Proof of Unlearning 

 
 As mentioned previously, membership inference attacks serve to identify information leakage within unlearning systems, while backdoor attacks aim to implant backdoors in models, leading to the models incorrectly classifying certain inputs when specific triggers are present. Viewing from a different angle, in a machine unlearning setting, if unlearned data show no membership in the model post-unlearning, or if backdoors within the unlearned data are undetectable, it is logical to conclude that the unlearned data has been effectively removed from the model with high probability. Based on this rationale, many studies on unlearning methodologies have employed the standard form or variants of inference attacks [ 125 , 126 , 127 , 128 , 129 , 130 , 131 ] and backdoor attacks [ 132 , 57 , 133 , 134 , 135 , 136 ] as tools serving as proof of unlearning. Some other studies utilize adversarial attacks to detect the insufficiency of unlearning procedures [ 137 , 138 , 102 , 103 ] .
This list represents only a selection of studies, as the use of attacks to demonstrate unlearning has been thoroughly reviewed in prior literature. For those seeking more in-depth information on this field, we recommend referring to [ 4 , 13 , 19 , 31 , 32 ] for comprehensive details.

 
 
 Fig. 5 : Threats, attacks, and defenses in machine unlearning. 
 
 
 
 

## IX Summary and Discussion 

 
 So far, we have reviewed the related works on threats in MU systems, along with an analysis of various attack types, defense mechanisms, and their interactions in different roles. To clarify these relationships, as illustrated in Figure 5 , this section offers a summary and discussion.

 
 
 Machine unlearning. MU presents both threats and opportunities for defense: (i) the process of machine unlearning and the handling of unlearning requests introduce threats such as information leakage and the potential for malicious unlearning attacks, and (ii) unlearning can be used as a defensive strategy for model recovery and remove undesirable knowledge for value alignment.

 
 
 Threats. Threats arise from the unlearning process and the workflow of MU systems, creating opportunities for attacks. Therefore, corresponding defensive strategies must be explored and developed.

 
 
 Attacks. Attacks serve two key purposes: (i) to explore potential threats in MU systems, and (ii) to evaluate the performance of unlearning systems, including aspects such as privacy leakage, robustness, and effectiveness.

 
 
 Defenses. As mentioned earlier, (i) MU can serve as a defense in certain scenarios, and (ii) defenses against specific threats in MU systems still need to be thoroughly explored.

 
 
 

## X Challenges and Promising Directions 

 
 In earlier sections, we provided a detailed overview of threats, attacks, and defenses within machine unlearning systems. Yet, as unlearning methodologies continue to advance and their application expands, a range of new challenges and open problems emerge, necessitating further exploration. Consequently, this section aims to expand our discussion on these emerging challenges, suggesting potential directions for future research that could significantly improve the safety and reliability of machine unlearning systems.

 
 

### X-A Defenses against Malicious Unlearning 

 
 Note that one of the main threats to machine unlearning systems is the negative impact on unlearned models caused by malicious unlearning (see Section VI-B for more details). This threat manifests as either degraded model performance or the introduction of backdoors, involving the submission of malicious unlearning requests. While direct malicious unlearning requests can often be easily detected by checking if the requested data for removal is part of the training dataset [ 50 ] , identifying such requests during preconditioned unlearning attacks presents a nontrivial challenge. This is because the data requested for removal in these attacks are actually part of the training dataset and usually are mitigation data without adversarial features. Therefore, developing robust detection mechanisms for identifying these subtle and sophisticated unlearning requests becomes essential in safeguarding the integrity of machine unlearning systems.

 
 
 

### X-B Federated Unlearning 

 
 As highlighted in [ 25 , 26 ] , federated learning’s distinct features pose specific challenges to machine unlearning methodologies, largely due to knowledge diffusion through aggregation and data isolation aimed at protecting user data privacy. Consequently, threats in a federated unlearning context are inherently more complex. Additionally, the involvement of multiple users in federated unlearning means that attacks executed in a distributed fashion can be particularly stealthy and difficult to detect. However, this area remains largely unexplored, indicating a significant gap in current research.

 
 
 In addition, within a federated setting, users’ data are stored locally, prohibiting the transfer of local data to the server as it would violate the fundamental privacy guarantees of federated learning. Thus, understanding how unlearning requests are made, their specific format, and how these requests are managed by the server in a federated unlearning context is crucial. Studying these aspects is essential to developing an unlearning mechanism that adheres to both the privacy principles of federated learning and the compliance requirements of RTBF.

 
 
 

### X-C Privacy Preservation 

 
 Current machine unlearning systems operate under the assumption that the data designated for removal is accessible to the server responsible for the unlearning process. However, in an MLaaS setting, the model developer and service provider may be distinct entities [ 50 ] . In such scenarios, the service provider should not access the user’s data, raising questions about how to protect the privacy of the unlearned data. This challenge echoes the privacy concerns highlighted in the federated learning context.

 
 
 In general, one could employ Privacy-Enhancing Technologies (PETs) like homomorphic encryption, secure multi-party computation, or differential privacy to facilitate privacy-preserving machine learning [ 139 , 140 , 141 , 142 , 143 , 144 , 145 ] . Nevertheless, given that machine unlearning methodologies significantly diverge from traditional machine learning practices and introduce new privacy concerns, achieving privacy-preserving machine unlearning presents new challenges. Although there has been some investigation into privacy-preserving machine unlearning [ 146 , 147 ] , the trade-offs introduced by these technologies between privacy protection, unlearning efficiency, and model performance in the context of MU remain largely unexplored.

 
 
 

### X-D Unlearning for Large Models 

 
 Investigating the unlearning for large models is an emerging and vital area of research, crucial for enhancing the security and robustness of large model systems. This exploration faces several challenges, including:

 
 • 
 
 The inefficiency of the unlearning for large models, which necessitates innovative solutions to improve its effectiveness
 [ 35 ] .

 

 • 
 
 The difficulty in verifying unlearning effectiveness in large models, as evidenced by [ 108 ] , suggests that methods like MIA used for proving unlearning are less effective in larger models.

 

 • 
 
 The inherent non-explainability of large models complicates understanding the unlearning process. For instance, [ 72 ] demonstrates that unlearned information can still be retrieved through embedding attacks after unlearning.

 

 • 
 
 The exploration of threats and attacks specific to unlearning in large models is still lacking.

 

 
 
 
 Given the widespread deployment of large models in numerous applications [ 148 , 149 , 150 , 151 , 152 , 153 , 154 ] , tackling these challenges and intensifying research on threats, attacks, and defenses within large model unlearning systems is imperative.

 
 
 
 

## XI Conclusions 

 
 This survey examines machine unlearning as a vital component of efforts towards achieving safe AI, concentrating on threats and attacks against these systems, alongside the defensive strategies employed to counteract them. It provides a detailed analysis of methodologies and creates a comprehensive taxonomy based on their threat models. Furthermore, the survey investigates how unlearning can act as a defense in different scenarios and how attacks can serve as effective tools for testing and evaluating unlearning systems. Lastly, this work identifies challenges and outlines future directions for improving the safety, reliability, and privacy compliance of machine unlearning, stressing the critical need for continued exploration in this area.

 
 
 

## References

 
 
 [1] 
 
G. D. P. Regulation, “General data protection regulation (gdpr),” Intersoft Consulting, Accessed in October , vol. 24, no. 1, 2018.

 

 
 [2] 
 
H. Iwase, “Overview of the act on the protection of personal information,” Eur. Data Prot. L. Rev. , vol. 5, p. 92, 2019.

 

 
 [3] 
 
E. Goldman, “An introduction to the california consumer privacy act (ccpa),” Santa Clara Univ. Legal Studies Research Paper , 2020.

 

 
 [4] 
 
H. Xu, T. Zhu, L. Zhang, W. Zhou, and P. S. Yu, “Machine unlearning: A survey,” ACM Computing Surveys , vol. 56, no. 1, pp. 1–36, 2023.

 

 
 [5] 
 
Y. Cao and J. Yang, “Towards making systems forget with machine unlearning,” in 2015 IEEE symposium on security and privacy . IEEE, 2015, pp. 463–480.

 

 
 [6] 
 
A. Warnecke, L. Pirch, C. Wressnegger, and K. Rieck, “Machine unlearning of features and labels,” arXiv preprint arXiv:2108.11577 , 2021.

 

 
 [7] 
 
Y. Zhao, P. Wang, H. Qi, J. Huang, Z. Wei, and Q. Zhang, “Federated unlearning with momentum degradation,” IEEE Internet of Things Journal , 2023.

 

 
 [8] 
 
Y. Liu, L. Xu, X. Yuan, C. Wang, and B. Li, “The right to be forgotten in federated learning: An efficient realization with rapid retraining,” in IEEE INFOCOM 2022-IEEE Conference on Computer Communications . IEEE, 2022, pp. 1749–1758.

 

 
 [9] 
 
P. Wang, W. Song, H. Qi, C. Zhou, F. Li, Y. Wang, P. Sun, and Q. Zhang, “Server-initiated federated unlearning to eliminate impacts of low-quality data,” IEEE Transactions on Services Computing , no. 01, pp. 1–15, 2024.

 

 
 [10] 
 
H. Hu, S. Wang, T. Dong, and M. Xue, “Learn what you want to unlearn: Unlearning inversion attacks against machine unlearning,” in 2024 IEEE Symposium on Security and Privacy (SP) , 2024.

 

 
 [11] 
 
T. T. Nguyen, T. T. Huynh, P. L. Nguyen, A. W.-C. Liew, H. Yin, and Q. V. H. Nguyen, “A survey of machine unlearning,” arXiv preprint arXiv:2209.02299 , 2022.

 

 
 [12] 
 
J. Xu, Z. Wu, C. Wang, and X. Jia, “Machine unlearning: Solutions and challenges,” arXiv preprint arXiv:2308.07061 , 2023.

 

 
 [13] 
 
Y. Qu, X. Yuan, M. Ding, W. Ni, T. Rakotoarivelo, and D. Smith, “Learn to unlearn: A survey on machine unlearning,” arXiv preprint arXiv:2305.07512 , 2023.

 

 
 [14] 
 
Y. Lin, Z. Gao, H. Du, J. Ren, Z. Xie, and D. Niyato, “Blockchain-enabled trustworthy federated unlearning,” arXiv preprint arXiv:2401.15917 , 2024.

 

 
 [15] 
 
H. Qiu, Y. Wang, Y. Xu, L. Cui, and Z. Shen, “Fedcio: Efficient exact federated unlearning with clustering, isolation, and one-shot aggregation,” in 2023 IEEE International Conference on Big Data (BigData) . IEEE, 2023, pp. 5559–5568.

 

 
 [16] 
 
N. Ding, E. Wei, and R. Berry, “Strategic data revocation in federated unlearning,” arXiv preprint arXiv:2312.01235 , 2023.

 

 
 [17] 
 
J. Shao, T. Lin, X. Cao, and B. Luo, “Federated unlearning: a perspective of stability and fairness,” arXiv preprint arXiv:2402.01276 , 2024.

 

 
 [18] 
 
N. Ding, Z. Sun, E. Wei, and R. Berry, “Incentivized federated learning and unlearning,” arXiv preprint arXiv:2308.12502 , 2023.

 

 
 [19] 
 
T. Shaik, X. Tao, H. Xie, L. Li, X. Zhu, and Q. Li, “Exploring the landscape of machine unlearning: A survey and taxonomy,” arXiv preprint arXiv:2305.06360 , 2023.

 

 
 [20] 
 
N. Ding, Z. Sun, E. Wei, and R. Berry, “Incentive mechanism design for federated learning and unlearning,” in Proceedings of the Twenty-fourth International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing , 2023, pp. 11–20.

 

 
 [21] 
 
Y. Lin, Z. Gao, H. Du, D. Niyato, G. Gui, S. Cui, and J. Ren, “Scalable federated unlearning via isolated and coded sharding,” arXiv preprint arXiv:2401.15957 , 2024.

 

 
 [22] 
 
N. Li, C. Zhou, Y. Gao, H. Chen, A. Fu, Z. Zhang, and Y. Shui, “Machine unlearning: Taxonomy, metrics, applications, challenges, and prospects,” arXiv preprint arXiv:2403.08254 , 2024.

 

 
 [23] 
 
H. Liu, P. Xiong, T. Zhu, and P. S. Yu, “A survey on machine unlearning: Techniques and new emerged privacy risks,” arXiv preprint arXiv:2406.06186 , 2024.

 

 
 [24] 
 
W. Wang, Z. Tian, and S. Yu, “Machine unlearning: A comprehensive survey,” arXiv preprint arXiv:2405.07406 , 2024.

 

 
 [25] 
 
Z. Liu, Y. Jiang, J. Shen, M. Peng, K.-Y. Lam, X. Yuan, and X. Liu, “A survey on federated unlearning: Challenges, methods, and future directions,” ACM Computing Surveys , 2023.

 

 
 [26] 
 
H. Jeong, S. Ma, and A. Houmansadr, “Sok: Challenges and opportunities in federated unlearning,” arXiv preprint arXiv:2403.02437 , 2024.

 

 
 [27] 
 
N. Romandini, A. Mora, C. Mazzocca, R. Montanari, and P. Bellavista, “Federated unlearning: A survey on methods, design guidelines, and evaluation metrics,” arXiv preprint arXiv:2401.05146 , 2024.

 

 
 [28] 
 
J. Yang and Y. Zhao, “A survey of federated unlearning: A taxonomy, challenges and future directions,” arXiv preprint arXiv:2310.19218 , 2023.

 

 
 [29] 
 
F. Wang, B. Li, and B. Li, “Federated unlearning and its privacy threats,” IEEE Network , 2023.

 

 
 [30] 
 
N. Li, A. Pan, A. Gopal, S. Yue, D. Berrios, A. Gatti, J. D. Li, A.-K. Dombrowski, S. Goel, L. Phan et al. , “The wmdp benchmark: Measuring and reducing malicious use with unlearning,” arXiv preprint arXiv:2403.03218 , 2024.

 

 
 [31] 
 
A. Lynch, P. Guo, A. Ewart, S. Casper, and D. Hadfield-Menell, “Eight methods to evaluate robust unlearning in llms,” arXiv preprint arXiv:2402.16835 , 2024.

 

 
 [32] 
 
P. Thaker, Y. Maurya, and V. Smith, “Guardrail baselines for unlearning in llms,” arXiv preprint arXiv:2403.03329 , 2024.

 

 
 [33] 
 
N. Si, H. Zhang, H. Chang, W. Zhang, D. Qu, and W. Zhang, “Knowledge unlearning for llms: Tasks, methods, and challenges,” arXiv preprint arXiv:2311.15766 , 2023.

 

 
 [34] 
 
S. Liu, Y. Yao, J. Jia, S. Casper, N. Baracaldo, P. Hase, X. Xu, Y. Yao, H. Li, K. R. Varshney et al. , “Rethinking machine unlearning for large language models,” arXiv preprint arXiv:2402.08787 , 2024.

 

 
 [35] 
 
Z. Liu, G. Dou, Z. Tan, Y. Tian, and M. Jiang, “Machine unlearning in generative ai: A survey,” arXiv preprint arXiv:2407.20516 , 2024.

 

 
 [36] 
 
M. Ribeiro, K. Grolinger, and M. A. Capretz, “Mlaas: Machine learning as a service,” in 2015 IEEE 14th international conference on machine learning and applications (ICMLA) . IEEE, 2015, pp. 896–902.

 

 
 [37] 
 
E. Hesamifard, H. Takabi, M. Ghasemi, and R. N. Wright, “Privacy-preserving machine learning as a service,” Proceedings on Privacy Enhancing Technologies , 2018.

 

 
 [38] 
 
Z. Liu, I. Tjuawinata, C. Xing, and K.-Y. Lam, “Mpc-enabled privacy-preserving neural network training against malicious attack,” arXiv preprint arXiv:2007.12557 , 2020.

 

 
 [39] 
 
Q. Weng, W. Xiao, Y. Yu, W. Wang, C. Wang, J. He, Y. Li, L. Zhang, W. Lin, and Y. Ding, “ { \{ MLaaS } \} in the wild: Workload analysis and scheduling in { \{ Large-Scale } \} heterogeneous { \{ GPU } \} clusters,” in 19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22) , 2022, pp. 945–960.

 

 
 [40] 
 
H. Zhang, Y. Li, Y. Huang, Y. Wen, J. Yin, and K. Guan, “Mlmodelci: An automatic cloud platform for efficient mlaas,” in Proceedings of the 28th ACM International Conference on Multimedia , 2020, pp. 4453–4456.

 

 
 [41] 
 
H. Yu, K. Yang, T. Zhang, Y.-Y. Tsai, T.-Y. Ho, and Y. Jin, “Cloudleak: Large-scale deep learning models stealing through adversarial examples.” in NDSS , vol. 38, 2020, p. 102.

 

 
 [42] 
 
M. Kesarwani, B. Mukhoty, V. Arya, and S. Mehta, “Model extraction warning in mlaas paradigm,” in Proceedings of the 34th Annual Computer Security Applications Conference , 2018, pp. 371–380.

 

 
 [43] 
 
A. Chakraborty, M. Alam, V. Dey, A. Chattopadhyay, and D. Mukhopadhyay, “Adversarial attacks and defences: A survey,” arXiv preprint arXiv:1810.00069 , 2018.

 

 
 [44] 
 
Z. Tian, L. Cui, J. Liang, and S. Yu, “A comprehensive survey on poisoning attacks and countermeasures in machine learning,” ACM Computing Surveys , vol. 55, no. 8, pp. 1–35, 2022.

 

 
 [45] 
 
M. Fredrikson, S. Jha, and T. Ristenpart, “Model inversion attacks that exploit confidence information and basic countermeasures,” in Proceedings of the 22nd ACM SIGSAC conference on computer and communications security , 2015, pp. 1322–1333.

 

 
 [46] 
 
R. Shokri, M. Stronati, C. Song, and V. Shmatikov, “Membership inference attacks against machine learning models,” in 2017 IEEE symposium on security and privacy (SP) . IEEE, 2017, pp. 3–18.

 

 
 [47] 
 
Y. Li, Y. Jiang, Z. Li, and S.-T. Xia, “Backdoor learning: A survey,” IEEE Transactions on Neural Networks and Learning Systems , vol. 35, no. 1, pp. 5–22, 2022.

 

 
 [48] 
 
T. Gu, B. Dolan-Gavitt, and S. Garg, “Badnets: Identifying vulnerabilities in the machine learning model supply chain,” arXiv preprint arXiv:1708.06733 , 2017.

 

 
 [49] 
 
S. Li, M. Xue, B. Z. H. Zhao, H. Zhu, and X. Zhang, “Invisible backdoor attacks on deep neural networks via steganography and regularization,” IEEE Transactions on Dependable and Secure Computing , vol. 18, no. 5, pp. 2088–2105, 2020.

 

 
 [50] 
 
H. Hu, S. Wang, J. Chang, H. Zhong, R. Sun, S. Hao, H. Zhu, and M. Xue, “A duty to forget, a right to be assured? exposing vulnerabilities in machine unlearning services,” in Proceedings of the Network and Distributed System Security Symposium , 2024.

 

 
 [51] 
 
Y. Hu, J. Lou, J. Liu, F. Lin, Z. Qin, and K. Ren, “Eraser: Machine unlearning in mlaas via an inference serving-aware approach,” arXiv preprint arXiv:2311.16136 , 2023.

 

 
 [52] 
 
J. Z. Di, J. Douglas, J. Acharya, G. Kamath, and A. Sekhari, “Hidden poison: Machine unlearning enables camouflaged poisoning attacks,” in NeurIPS ML Safety Workshop , 2022.

 

 
 [53] 
 
Y. Xie, J. Zhang, S. Zhao, T. Zhang, and X. Chen, “Same: Sample reconstruction against model extraction attacks,” arXiv preprint arXiv:2312.10578 , 2023.

 

 
 [54] 
 
Z. He, T. Zhang, and R. B. Lee, “Model inversion attacks against collaborative inference,” in Proceedings of the 35th Annual Computer Security Applications Conference , 2019, pp. 148–162.

 

 
 [55] 
 
M. Chen, Z. Zhang, T. Wang, M. Backes, M. Humbert, and Y. Zhang, “When machine unlearning jeopardizes privacy,” in Proceedings of the 2021 ACM SIGSAC conference on computer and communications security , 2021, pp. 896–911.

 

 
 [56] 
 
Y. Liu, S. Ma, Y. Aafer, W.-C. Lee, J. Zhai, W. Wang, and X. Zhang, “Trojaning attack on neural networks,” in 25th Annual Network And Distributed System Security Symposium (NDSS 2018) . Internet Soc, 2018.

 

 
 [57] 
 
D. M. Sommer, L. Song, S. Wagh, and P. Mittal, “Towards probabilistic verification of machine unlearning,” arXiv preprint arXiv:2003.04247 , 2020.

 

 
 [58] 
 
C. Dwork, “Differential privacy,” in International colloquium on automata, languages, and programming . Springer, 2006, pp. 1–12.

 

 
 [59] 
 
K. Zhang, Y. Zhang, R. Sun, P.-W. Tsai, M. U. Hassan, X. Yuan, M. Xue, and J. Chen, “Bounded and unbiased composite differential privacy,” in 2024 IEEE Symposium on Security and Privacy (SP) , 2024.

 

 
 [60] 
 
P. Zhang, J. Sun, M. Tan, and X. Wang, “Exploiting machine unlearning for backdoor attacks in deep learning system,” arXiv preprint arXiv:2310.10659 , 2023.

 

 
 [61] 
 
Y. Liu, M. Fan, C. Chen, X. Liu, Z. Ma, L. Wang, and J. Ma, “Backdoor defense with machine unlearning,” in IEEE INFOCOM 2022-IEEE Conference on Computer Communications . IEEE, 2022, pp. 280–289.

 

 
 [62] 
 
X. Cao, J. Jia, Z. Zhang, and N. Z. Gong, “Fedrecover: Recovering from poisoning attacks in federated learning using historical information,” in 2023 IEEE Symposium on Security and Privacy (SP) . IEEE, 2023, pp. 1366–1383.

 

 
 [63] 
 
B. Wang, Y. Yao, S. Shan, H. Li, B. Viswanath, H. Zheng, and B. Y. Zhao, “Neural cleanse: Identifying and mitigating backdoor attacks in neural networks,” in 2019 IEEE Symposium on Security and Privacy (SP) . IEEE, 2019, pp. 707–723.

 

 
 [64] 
 
X. Lu, S. Welleck, J. Hessel, L. Jiang, L. Qin, P. West, P. Ammanabrolu, and Y. Choi, “Quark: Controllable text generation with reinforced unlearning,” Advances in neural information processing systems , vol. 35, pp. 27 591–27 609, 2022.

 

 
 [65] 
 
J. Liu, P. Ram, Y. Yao, G. Liu, Y. Liu, P. SHARMA, S. Liu et al. , “Model sparsity can simplify machine unlearning,” Advances in Neural Information Processing Systems , vol. 36, 2024.

 

 
 [66] 
 
Z. Lu, H. Liang, M. Zhao, Q. Lv, T. Liang, and Y. Wang, “Label-only membership inference attacks on machine unlearning without dependence of posteriors,” International Journal of Intelligent Systems , vol. 37, no. 11, pp. 9424–9441, 2022.

 

 
 [67] 
 
J. Gao, S. Garg, M. Mahmoody, and P. N. Vasudevan, “Deletion inference, reconstruction, and compliance in machine (un) learning,” Proceedings on Privacy Enhancing Technologies , 2022.

 

 
 [68] 
 
J. Geiping, H. Bauermeister, H. Dröge, and M. Moeller, “Inverting gradients-how easy is it to break privacy in federated learning?” Advances in neural information processing systems , vol. 33, pp. 16 937–16 947, 2020.

 

 
 [69] 
 
L. Zhu, Z. Liu, and S. Han, “Deep leakage from gradients,” Advances in Neural Information Processing Systems , vol. 32, 2019.

 

 
 [70] 
 
J. Du, Z. Wang, and K. Ren, “Textual unlearning gives a false sense of unlearning,” arXiv preprint arXiv:2406.13348 , 2024.

 

 
 [71] 
 
R. Chourasia and N. Shah, “Forget unlearning: Towards true data-deletion in machine learning,” in International Conference on Machine Learning . PMLR, 2023, pp. 6028–6073.

 

 
 [72] 
 
L. Schwinn, D. Dobre, S. Xhonneux, G. Gidel, and S. Gunnemann, “Soft prompt threats: Attacking safety alignment and unlearning in open-source llms through the embedding space,” arXiv preprint arXiv:2402.09063 , 2024.

 

 
 [73] 
 
V. Gupta, C. Jung, S. Neel, A. Roth, S. Sharifi-Malvajerdi, and C. Waites, “Adaptive machine unlearning,” Advances in Neural Information Processing Systems , vol. 34, pp. 16 319–16 330, 2021.

 

 
 [74] 
 
C. Zhao, W. Qian, R. Ying, and M. Huai, “Static and sequential malicious attacks in the context of selective forgetting,” Advances in Neural Information Processing Systems , vol. 36, 2024.

 

 
 [75] 
 
N. Carlini and D. Wagner, “Towards evaluating the robustness of neural networks,” in 2017 ieee symposium on security and privacy (sp) . Ieee, 2017, pp. 39–57.

 

 
 [76] 
 
P.-Y. Chen, H. Zhang, Y. Sharma, J. Yi, and C.-J. Hsieh, “Zoo: Zeroth order optimization based black-box attacks to deep neural networks without training substitute models,” in Proceedings of the 10th ACM workshop on artificial intelligence and security , 2017, pp. 15–26.

 

 
 [77] 
 
W. Qian, C. Zhao, W. Le, M. Ma, and M. Huai, “Towards understanding and enhancing robustness of deep learning models against malicious unlearning attacks,” in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , 2023, pp. 1932–1942.

 

 
 [78] 
 
J. Geiping, L. H. Fowl, W. R. Huang, W. Czaja, G. Taylor, M. Moeller, and T. Goldstein, “Witches’ brew: Industrial scale data poisoning via gradient matching,” in International Conference on Learning Representations , 2020.

 

 
 [79] 
 
T. Gu, K. Liu, B. Dolan-Gavitt, and S. Garg, “Badnets: Evaluating backdooring attacks on deep neural networks,” IEEE Access , vol. 7, pp. 47 230–47 244, 2019.

 

 
 [80] 
 
Z. Huang, Y. Mao, and S. Zhong, “ { \{ UBA-Inf } \} : Unlearning activated backdoor attack with { \{ Influence-Driven } \} camouflage,” in 33rd USENIX Security Symposium (USENIX Security 24) , 2024, pp. 4211–4228.

 

 
 [81] 
 
B. Ma, T. Zheng, H. Hu, D. Wang, S. Wang, Z. Ba, Z. Qin, and K. Ren, “Releasing malevolence from benevolence: The menace of benign data on machine unlearning,” arXiv preprint arXiv:2407.05112 , 2024.

 

 
 [82] 
 
N. G. Marchant, B. I. Rubinstein, and S. Alfeld, “Hard to forget: Poisoning attacks on certified machine unlearning,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 36, no. 7, 2022, pp. 7691–7700.

 

 
 [83] 
 
C. Guo, T. Goldstein, A. Hannun, and L. Van Der Maaten, “Certified data removal from machine learning models,” arXiv preprint arXiv:1911.03030 , 2019.

 

 
 [84] 
 
S. Mei and X. Zhu, “Using machine teaching to identify optimal training-set attacks on machine learners,” in Proceedings of the aaai conference on artificial intelligence , vol. 29, no. 1, 2015.

 

 
 [85] 
 
S. R. Kadhe, A. Halimi, A. Rawat, and N. Baracaldo, “Fairsisa: Ensemble post-processing to improve fairness of unlearning in llms,” arXiv preprint arXiv:2312.07420 , 2023.

 

 
 [86] 
 
L. Bourtoule, V. Chandrasekaran, C. A. Choquette-Choo, H. Jia, A. Travers, B. Zhang, D. Lie, and N. Papernot, “Machine unlearning,” in 2021 IEEE Symposium on Security and Privacy (SP) . IEEE, 2021, pp. 141–159.

 

 
 [87] 
 
I. B. Soares, D. Wei, K. N. Ramamurthy, M. Singh, and M. Yurochkin, “Your fairness may vary: pretrained language model fairness in toxic text classification,” in Annual Meeting of the Association for Computational Linguistics , 2022.

 

 
 [88] 
 
J. Tan, F. Sun, R. Qiu, D. Su, and H. Shen, “Unlink to unlearn: Simplifying edge unlearning in gnns,” arXiv preprint arXiv:2402.10695 , 2024.

 

 
 [89] 
 
J. Cheng, G. Dasoulas, H. He, C. Agarwal, and M. Zitnik, “Gnndelete: A general strategy for unlearning in graph neural networks,” arXiv preprint arXiv:2302.13406 , 2023.

 

 
 [90] 
 
M. Bertran, S. Tang, M. Kearns, J. Morgenstern, A. Roth, and Z. S. Wu, “Reconstruction attacks on machine unlearning: Simple models are vulnerable,” arXiv preprint arXiv:2405.20272 , 2024.

 

 
 [91] 
 
Z. Liu, T. Wang, M. Huai, and C. Miao, “Backdoor attacks via machine unlearning,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 38, no. 13, 2024, pp. 14 115–14 123.

 

 
 [92] 
 
S. Li, H. Liu, T. Dong, B. Z. H. Zhao, M. Xue, H. Zhu, and J. Lu, “Hidden backdoors in human-centric language models,” in Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security , 2021, pp. 3123–3140.

 

 
 [93] 
 
J. Zhang, C. Dongdong, Q. Huang, J. Liao, W. Zhang, H. Feng, G. Hua, and N. Yu, “Poison ink: Robust and invisible backdoor attack,” IEEE Transactions on Image Processing , vol. 31, pp. 5691–5705, 2022.

 

 
 [94] 
 
X. Han, Y. Wu, Q. Zhang, Y. Zhou, Y. Xu, H. Qiu, G. Xu, and T. Zhang, “Backdooring multimodal learning,” in 2024 IEEE Symposium on Security and Privacy (SP) . IEEE Computer Society, 2023, pp. 31–31.

 

 
 [95] 
 
Y. Wu, J. Zhang, F. Kerschbaum, T. Zhang et al. , “Backdooring textual inversion for concept censorship,” arXiv preprint arXiv:2308.10718 , 2023.

 

 
 [96] 
 
H. Wang, T. Xiang, S. Guo, J. He, H. Liu, and T. Zhang, “Transtroj: Transferable backdoor attacks to pre-trained models via embedding indistinguishability,” arXiv preprint arXiv:2401.15883 , 2024.

 

 
 [97] 
 
J. Xu, M. Xue, and S. Picek, “Explainability-based backdoor attacks against graph neural networks,” in Proceedings of the 3rd ACM workshop on wireless security and machine learning , 2021, pp. 31–36.

 

 
 [98] 
 
W. Ma, D. Wang, R. Sun, M. Xue, S. Wen, and Y. Xiang, “The” beatrix”resurrections: Robust backdoor detection via gram matrices,” in Proceedings of the Network And Distributed System Security Symposium (NDSS 2023) , 2023.

 

 
 [99] 
 
Y. Li, H. Ma, Z. Zhang, Y. Gao, A. Abuadbba, M. Xue, A. Fu, Y. Zheng, S. F. Al-Sarawi, and D. Abbott, “Ntd: Non-transferability enabled deep learning backdoor detection,” IEEE Transactions on Information Forensics and Security , 2023.

 

 
 [100] 
 
W. Guo, L. Wang, X. Xing, M. Du, and D. Song, “Tabor: A highly accurate approach to inspecting and restoring trojan backdoors in ai systems,” arXiv preprint arXiv:1908.01763 , 2019.

 

 
 [101] 
 
S. Goel, A. Prabhu, P. Torr, P. Kumaraguru, and A. Sanyal, “Corrective machine unlearning,” arXiv preprint arXiv:2402.14015 , 2024.

 

 
 [102] 
 
Y. Zeng, S. Chen, W. Park, Z. M. Mao, M. Jin, and R. Jia, “Adversarial unlearning of backdoors via implicit hypergradient,” arXiv preprint arXiv:2110.03735 , 2021.

 

 
 [103] 
 
S. Wei, M. Zhang, H. Zha, and B. Wu, “Shared adversarial unlearning: Backdoor mitigation by unlearning shared adversarial examples,” Advances in Neural Information Processing Systems , vol. 36, 2024.

 

 
 [104] 
 
H. Bansal, N. Singhi, Y. Yang, F. Yin, A. Grover, and K.-W. Chang, “Cleanclip: Mitigating data poisoning attacks in multimodal contrastive learning,” arXiv preprint arXiv:2303.03323 , 2023.

 

 
 [105] 
 
Y. Jiang, J. Shen, Z. Liu, C. W. Tan, and K.-Y. Lam, “Towards efficient and certified recovery from poisoning attacks in federated learning,” arXiv preprint arXiv:2401.08216 , 2024.

 

 
 [106] 
 
Y. Li, C. Chen, X. Zheng, Y. Zhang, Z. Han, D. Meng, and J. Wang, “Making users indistinguishable: Attribute-wise unlearning in recommender systems,” in Proceedings of the 31st ACM International Conference on Multimedia , 2023, pp. 984–994.

 

 
 [107] 
 
Y. Xue, J. Liu, S. McDonagh, and S. A. Tsaftaris, “Erase to enhance: Data-efficient machine unlearning in mri reconstruction,” in Medical Imaging with Deep Learning , 2024.

 

 
 [108] 
 
Y. Yao, X. Xu, and Y. Liu, “Large language model unlearning,” arXiv preprint arXiv:2310.10683 , 2023.

 

 
 [109] 
 
H. Li, G. Deng, Y. Liu, K. Wang, Y. Li, T. Zhang, Y. Liu, G. Xu, G. Xu, and H. Wang, “Digger: Detecting copyright content mis-usage in large language model training,” arXiv preprint arXiv:2401.00676 , 2024.

 

 
 [110] 
 
Z. Liu, G. Dou, Z. Tan, Y. Tian, and M. Jiang, “Towards safer large language models through machine unlearning,” arXiv preprint arXiv:2402.10058 , 2024.

 

 
 [111] 
 
P. Wang, Z. Wei, H. Qi, S. Wan, Y. Xiao, G. Sun, and Q. Zhang, “Mitigating poor data quality impact with federated unlearning for human-centric metaverse,” IEEE Journal on Selected Areas in Communications , 2023.

 

 
 [112] 
 
X. Guo, P. Wang, S. Qiu, W. Song, Q. Zhang, X. Wei, and D. Zhou, “Fast: Adopting federated unlearning to eliminating malicious terminals at server side,” IEEE Transactions on Network Science and Engineering , 2023.

 

 
 [113] 
 
H. Bano, M. Ameen, M. Mehdi, A. Hussain, and P. Wang, “Federated unlearning and server right to forget: Handling unreliable client contributions,” in International Conference on Recent Trends in Image Processing and Pattern Recognition . Springer, 2023, pp. 393–410.

 

 
 [114] 
 
M. Isonuma and I. Titov, “Unlearning reveals the influential training data of language models,” arXiv preprint arXiv:2401.15241 , 2024.

 

 
 [115] 
 
R. Sun, M. Xue, G. Tyson, S. Wang, S. Camtepe, and S. Nepal, “Not seen, not heard in the digital world! measuring privacy practices in children’s apps,” in Proceedings of the ACM Web Conference 2023 , 2023, pp. 2166–2177.

 

 
 [116] 
 
A. Hu, Z. Lu, R. Xie, and M. Xue, “Veridip: Verifying ownership of deep neural networks through privacy leakage fingerprints,” IEEE Transactions on Dependable and Secure Computing , 2023.

 

 
 [117] 
 
L. Wang, X. Zeng, J. Guo, K.-F. Wong, and G. Gottlob, “Selective forgetting: Advancing machine unlearning techniques and evaluation in language models,” arXiv preprint arXiv:2402.05813 , 2024.

 

 
 [118] 
 
Y. Cao, X. Xiao, R. Sun, D. Wang, M. Xue, and S. Wen, “Stylefool: Fooling video classification systems via style transfer,” in 2023 IEEE Symposium on Security and Privacy (SP) . IEEE, 2023, pp. 1631–1648.

 

 
 [119] 
 
Y. Cao, Z. Zhao, X. Xiao, D. Wang, M. Xue, and J. Lu, “Logostylefool: Vitiating video recognition systems via logo style transfer,” in 38th AAAI Conference on Artificial Intelligence . AAAI, 2024.

 

 
 [120] 
 
Y. Liu, J. Peng, J. James, and Y. Wu, “Ppgan: Privacy-preserving generative adversarial network,” in 2019 IEEE 25Th international conference on parallel and distributed systems (ICPADS) . IEEE, 2019, pp. 985–989.

 

 
 [121] 
 
Y. Cao, J. Li, X. Xiao, D. Wang, M. Xue, H. Ge, W. Liu, and G. Hu, “Localstylefool: Regional video style transfer attack using segment anything model,” in 2024 IEEE Symposium on Security and Privacy Workshop (SPW) , 2024.

 

 
 [122] 
 
Z. Zhang, Y. Zhou, X. Zhao, T. Che, and L. Lyu, “Prompt certified machine unlearning with randomized gradient smoothing and quantization,” Advances in Neural Information Processing Systems , vol. 35, pp. 13 433–13 455, 2022.

 

 
 [123] 
 
T. Che, Y. Zhou, Z. Zhang, L. Lyu, J. Liu, D. Yan, D. Dou, and J. Huan, “Fast federated machine unlearning with nonlinear functional theory,” in International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA , ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 4241–4268.

 

 
 [124] 
 
J. Cohen, E. Rosenfeld, and Z. Kolter, “Certified adversarial robustness via randomized smoothing,” in international conference on machine learning . PMLR, 2019, pp. 1310–1320.

 

 
 [125] 
 
V. S. Chundawat, A. K. Tarun, M. Mandal, and M. Kankanhalli, “Zero-shot machine unlearning,” IEEE Transactions on Information Forensics and Security , 2023.

 

 
 [126] 
 
Z. Ma, Y. Liu, X. Liu, J. Liu, J. Ma, and K. Ren, “Learn to forget: Machine unlearning via neuron masking,” IEEE Transactions on Dependable and Secure Computing , 2022.

 

 
 [127] 
 
J. Wang, S. Guo, X. Xie, and H. Qi, “Federated unlearning via class-discriminative pruning,” in Proceedings of the ACM Web Conference 2022 , 2022, pp. 622–632.

 

 
 [128] 
 
G. Liu, X. Ma, Y. Yang, C. Wang, and J. Liu, “Federaser: Enabling efficient client-level data removal from federated learning models,” in 2021 IEEE/ACM 29th International Symposium on Quality of Service (IWQOS) . IEEE, 2021, pp. 1–10.

 

 
 [129] 
 
Z. Liu, G. Dou, Y. Tian, C. Zhang, E. Chien, and Z. Zhu, “Breaking the trilemma of privacy, utility, efficiency via controllable machine unlearning,” arXiv preprint arXiv:2310.18574 , 2023.

 

 
 [130] 
 
A. Alag, Y. Huang, and K. Li, “Is ema robust? examining the robustness of data auditing and a novel non-calibration extension,” in NeurIPS 2023 Workshop on Regulatable ML , 2023.

 

 
 [131] 
 
D. Ye, T. Zhu, C. Zhu, D. Wang, S. Shen, W. Zhou et al. , “Reinforcement unlearning,” arXiv preprint arXiv:2312.15910 , 2023.

 

 
 [132] 
 
C. Wu, S. Zhu, and P. Mitra, “Federated unlearning with knowledge distillation. arxiv 2022,” arXiv preprint arXiv:2201.09441 .

 

 
 [133] 
 
Y. Guo, Y. Zhao, S. Hou, C. Wang, and X. Jia, “Verifying in the dark: Verifiable machine unlearning by using invisible backdoor triggers,” IEEE Transactions on Information Forensics and Security , 2023.

 

 
 [134] 
 
M. Pawelczyk, S. Neel, and H. Lakkaraju, “In-context unlearning: Language models as few shot unlearners,” arXiv preprint arXiv:2310.07579 , 2023.

 

 
 [135] 
 
A. Halimi, S. R. Kadhe, A. Rawat, and N. B. Angel, “Federated unlearning: How to efficiently erase a client in fl?” in International Conference on Machine Learning , 2022.

 

 
 [136] 
 
Y. Li, C. Chen, X. Zheng, and J. Zhang, “Federated unlearning via active forgetting,” arXiv preprint arXiv:2307.03363 , 2023.

 

 
 [137] 
 
Y. Zhang, J. Jia, X. Chen, A. Chen, Y. Zhang, J. Liu, K. Ding, and S. Liu, “To generate or not? safety-driven unlearned diffusion models are still easy to generate unsafe images… for now,” arXiv preprint arXiv:2310.11868 , 2023.

 

 
 [138] 
 
S. Goel, A. Prabhu, A. Sanyal, S.-N. Lim, P. Torr, and P. Kumaraguru, “Towards adversarial evaluations for inexact machine unlearning,” arXiv preprint arXiv:2201.06640 , 2022.

 

 
 [139] 
 
P. Mohassel and Y. Zhang, “Secureml: A system for scalable privacy-preserving machine learning,” in 2017 IEEE symposium on security and privacy (SP) . IEEE, 2017, pp. 19–38.

 

 
 [140] 
 
Z. Liu, J. Guo, K.-Y. Lam, and J. Zhao, “Efficient dropout-resilient aggregation for privacy-preserving machine learning,” IEEE Transactions on Information Forensics and Security , vol. 18, pp. 1839–1854, 2022.

 

 
 [141] 
 
Z. Liu, J. Guo, W. Yang, J. Fan, K.-Y. Lam, and J. Zhao, “Privacy-preserving aggregation in federated learning: A survey,” IEEE Transactions on Big Data , 2022.

 

 
 [142] 
 
K.-Y. Lam, X. Lu, L. Zhang, X. Wang, H. Wang, and S. Q. Goh, “Efficient fhe-based privacy-enhanced neural network for trustworthy ai-as-a-service,” IEEE Transactions on Dependable and Secure Computing , 2024.

 

 
 [143] 
 
Z. Liu, J. Guo, W. Yang, J. Fan, K.-Y. Lam, and J. Zhao, “Dynamic user clustering for efficient and privacy-preserving federated learning,” IEEE Transactions on Dependable and Secure Computing , 2024.

 

 
 [144] 
 
Z. Liu, H.-Y. Lin, and Y. Liu, “Long-term privacy-preserving aggregation with user-dynamics for federated learning,” IEEE Transactions on Information Forensics and Security , 2023.

 

 
 [145] 
 
S. Wagh, D. Gupta, and N. Chandran, “Securenn: 3-party secure computation for neural network training,” Proceedings on Privacy Enhancing Technologies , 2019.

 

 
 [146] 
 
Z. Liu, Y. Jiang, W. Jiang, J. Guo, J. Zhao, and K.-Y. Lam, “Guaranteeing data privacy in federated unlearning with dynamic user participation,” arXiv preprint arXiv:2406.00966 , 2024.

 

 
 [147] 
 
Z. Liu, H. Ye, Y. Jiang, J. Shen, J. Guo, I. Tjuawinata, and K.-Y. Lam, “Privacy-preserving federated unlearning with certified client removal,” arXiv preprint arXiv:2404.09724 , 2024.

 

 
 [148] 
 
S. Pan, L. Luo, Y. Wang, C. Chen, J. Wang, and X. Wu, “Unifying large language models and knowledge graphs: A roadmap,” IEEE Transactions on Knowledge and Data Engineering , 2024.

 

 
 [149] 
 
C. Chen, Y. Wang, Y. Zhang, Q. Z. Sheng, and K.-Y. Lam, “Separate-and-aggregate: A transformer-based patch refinement model for knowledge graph completion,” in International Conference on Advanced Data Mining and Applications . Springer, 2023, pp. 62–77.

 

 
 [150] 
 
E. Kasneci, K. Seßler, S. Küchemann, M. Bannert, D. Dementieva, F. Fischer, U. Gasser, G. Groh, S. Günnemann, E. Hüllermeier et al. , “Chatgpt for good? on opportunities and challenges of large language models for education,” Learning and individual differences , vol. 103, p. 102274, 2023.

 

 
 [151] 
 
C. Chen, Y. Wang, A. Sun, B. Li, and K.-Y. Lam, “Dipping plms sauce: Bridging structure and text for effective knowledge graph completion via conditional soft prompting,” arXiv preprint arXiv:2307.01709 , 2023.

 

 
 [152] 
 
A. J. Thirunavukarasu, D. S. J. Ting, K. Elangovan, L. Gutierrez, T. F. Tan, and D. S. W. Ting, “Large language models in medicine,” Nature medicine , vol. 29, no. 8, pp. 1930–1940, 2023.

 

 
 [153] 
 
C. Chen, Y. Wang, B. Li, and K.-Y. Lam, “Knowledge is flat: A seq2seq generative framework for various knowledge graph completion,” arXiv preprint arXiv:2209.07299 , 2022.

 

 
 [154] 
 
G. Deng, Y. Liu, Y. Li, K. Wang, Y. Zhang, Z. Li, H. Wang, T. Zhang, and Y. Liu, “Masterkey: Automated jailbreaking of large language model chatbots,” in Proc. ISOC NDSS , 2024.