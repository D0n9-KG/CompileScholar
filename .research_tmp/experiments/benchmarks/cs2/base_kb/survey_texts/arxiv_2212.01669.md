A Survey on Medical Document Summarization 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2212.01669v1 [cs.CL] 03 Dec 2022 
 
 

# A Survey on Medical Document Summarization

 DOI:  10.1145/1122445.1122456 Journal:  JACM Volume:  37 4 111 8 CCS:  Information systems Similarity measures CCS:  Information systems Information retrieval diversity CCS:  Information systems Combination, fusion and federated search CCS:  Information systems Language models CCS:  Information systems Top-k retrieval in databases CCS:  Information systems Speech / audio search CCS:  Information systems Video search CCS:  Information systems Image search CCS:  Information systems Retrieval efficiency CCS:  Information systems Summarization CCS:  Information systems Information extraction CCS:  Computing methodologies Neural networks CCS:  Computing methodologies Supervised learning CCS:  Computing methodologies Unsupervised learning CCS:  Computing methodologies Natural language generation CCS:  Computing methodologies Information extraction 
 
 
 Raghav Jain
 
 Affiliation:  Indian Institute of Technology Patna , Patna , Bihar , India 
 
 , 
 Anubhav Jangra
 
 
 
 email: anubhav0603@gmail.com 
 
 Affiliation:  Indian Institute of Technology Patna , India 
 
 , 
 Sriparna Saha
 
 Affiliation:  Indian Institute of Technology Patna , Patna , Bihar , India 
 
 and 
 Adam Jatowt
 
 Affiliation:  University of Innsbruck , Austria 
 
 2018 

 Abstract. 
 
 The internet has had a dramatic effect on the healthcare industry, allowing documents to be saved, shared, and managed digitally. This has made it easier to locate and share important data, improving patient care and providing more opportunities for medical studies. As there is so much data accessible to doctors and patients alike, summarizing it has become increasingly necessary - this has been supported through the introduction of deep learning and transformer-based networks, which have boosted the sector significantly in recent years. This paper gives a comprehensive survey of the current techniques and trends in medical summarization.

 
 
 
 Keywords:  summarization, clinical natural language processing, neural networks
 
 

## 1. Introduction

 
 The internet has become a global phenomenon, connecting people all over the world and allowing for the exchange of information on a scale that was previously unimaginable. The rise of the internet and the corresponding digitization of many aspects of daily life has had a profound impact on society leading to information overload ( Bontcheva et al., 2013 ) . The sheer amount of information available today can be overwhelming. To combat this, individuals can use summarization techniques to distill the information down to its most essential points. The internet also had a profound impact on medical science. With the proliferation of online health tools, it is now easier than ever before to access medical information and resources ( November, 2012 ) . For example, individuals can easily search for medical information, research medical conditions and treatments, and find healthcare providers. Additionally, social media platforms have provided a platform for medical professionals to collaborate, share information, and discuss current medical topics. This has allowed medical professionals to quickly access the latest research, treatments, and developments in the field. Furthermore, online tools and platforms have enabled medical professionals to perform remote consultations with patients, providing more efficient and convenient healthcare services. There are a few reasons why summarization is important for medical documents. First, it allows for a quick overview of the document’s content. This can be useful when trying to determine if the document is relevant to a particular topic of interest. Second, summarization can help to identify key points or ideas within a document. This can be valuable when trying to understand the main arguments or findings of a study. Finally, summarization can help to improve the readability of a document by reducing the amount of text that needs to be read. This application of summarization systems has the potential to reduce the burdens from medical workers who already are overburdened ( Portoghese et al., 2014 ) .

 
 
 Deep learning has been used in many other fields in addition to computer science ( Sarker, 2021 ) , especially medical science. Deep learning can be used to diagnose diseases ( Sharma et al., 2022 ) , predict patient outcomes ( Xie et al., 2019 ) , and even find new treatments ( Bian and Xie, 2021 ) . In addition, deep learning can be used to analyze medical images, such as X-rays and MRI scans ( Liu et al., 2021 ) . There are many different applications for deep learning in medical science, and the potential benefits are huge. Deep learning could potentially revolutionize medicine, and make it more effective and efficient. One of such application area that leveraged AI and deep learning immensely is Clinical Natural Language Processing (NLP) ( Wu et al., 2020 ) which can be defined as an interdisciplinary research field that involves the development of algorithms and systems to process natural language text from healthcare domains. It attempts to extract meaningful information from free-text clinical documents such as discharge summaries, clinical notes, and lab reports, in order to support clinical decision-making, clinical data mining, and other healthcare-related tasks. Clinical Natural language processing has gained a lot of popularity in the last few years due to its ability to enable better quality and cost-effective healthcare. The reason for this rise can be attributed to the release of large-scale datasets and different workshops ( Demner-Fushman et al., 2022 ) that are being organized to promote research in this area. Medical document summarization (MDS) can be considered as the subfield of clinical natural language processing. In the last few years, there have been various new advances in the field of medical document summarization, as illustrated in Figure 1 . This includes the introduction of new datasets, new methods for addressing the MDS problem, new challenges being organized in the community ( Ben Abacha et al., 2021 ; Nentidis et al., 2021 ) , and the introduction of more suitable evaluation metrics.

 
 
 At first, medical document summarization can be seen as a standard summarization problem only. However, after an extensive analysis, it can be observed that medical document summarization presents some unique and interesting challenges that may not be present in other domains because of the sensitivity of the medical domain and the complexity of the medical documents. These challenges expand the breadth of the problem, leading to a wider research scope for the task. The medical document summarization research field has seen a lot of progress in recent years. However, it is still disorganized compared to other areas in natural language processing. This lack of organization can make it difficult to gain an overall understanding of the research being conducted. There is currently no unified approach to this field, and research in this area continues to be fragmented. There is also lack of awareness about datasets available out there which leads to many works making their own datasets or not testing their approaches on other datasets. Additionally, resources that cover how to summarize medical documents and the types of medical documents available are not as readily available as they are in other fields. This restricts the exchange of ideas between different researchers. Furthermore, there has not been any standard evaluation set out there, so a conclusion on the best approach has not been possible. All of these factors work together to cause a lack of organization in the field of medical document summarization. This can make it difficult for researchers to keep track of the latest developments in the field. In addition, many of the papers published in this area are focused on specific applications of medical document summarization, such as health record summaries or research articles summaries. This makes it difficult to obtain an overall view of the state of the research field. These issues all together warrant the need for a survey on medical document summarization.

 
 
 The rest of the paper is structured as follows. We first discuss related works and surveys in Section 2 . We formally define the MDS task in Section 3 . In
Section 4 , we provide an extensive categorization of existing works based on different types of medical tasks. In Section 5 , we provide an categorization of existing works based on the type of input, output, and techniques used. In Section 6 , we discuss evaluation techniques devised for the evaluation of medical summaries. We then discuss the ethical concerns of deep learning while working in the medical domain in section 7 . We then provide a discussion based on all these in section 8 followed by possibilities of future work in Section 9 and conclude our paper in Section 10 .

 
 
 Figure 1. Trend in medical document summarization research over last two decades. X-axis: year, Y-axis: #papers on medical document summarization published in each year. The growing number of papers in the recent 3 years suggests that there should be more coming in the next years. 
 
 
 

## 2. Related Works

 
 Recent advances in deep learning, typically, with the introduction of the transformers ( Lewis et al., 2019 ; Raffel et al., 2019 ; Zhang et al., 2020b ) , have shown great success in text summarization leading to the study of different applications and domains within text summarization. With so much research going around summarization in the last few years, it is very difficult to be abreast with the current research and trends making it a necessity for researchers to work on survey papers around summarization so that community can catch up with this fast-moving area. A lot of application areas of summarization have been covered through different survey papers: Dialogue summarization ( Feng et al., 2021 ) , Multimodal summarization ( Jangra et al., 2021a ) , Video summarization ( Apostolidis et al., 2021 ) , multi-view video summarization ( Hussain et al., 2021 ) , multi-document summarization ( Ma et al., 2020 ) , multi-lingual summarization ( Wang et al., 2022 ) , source code summarization ( Zhang et al., 2022 ) etc. However, there is still a dearth of works focused on one of the most relevant use cases, i.e., medical document summarization. A thorough literature survey revealed that there is only one survey paper around medical document summarization ( Afantenos et al., 2005a ) that is from 2005. Apart from surveys about techniques for solving summarization, there is also a good amount of work around studying and comparing evaluation techniques ( Fabbri et al., 2020 ; Goel et al., 2021 ; Bhandari et al., 2020 ) .

 
 
 The medical domain has gained a lot of traction in the natural language processing community in the last few years. There are several surveys and comparative studies about the medical application of deep learning: medical imaging ( Pandey et al., 2021 ) , clinical NLP embeddings ( Kalyan and Sangeetha, 2020 ) , smart healthcare ( Zhou et al., 2021 ) , medical named entity recognition ( Bose et al., 2021 ) , medical question answering ( Jin et al., 2022 ) and medical dialogue ( Valizadeh and Parde, 2022 ) etc.

 
 
 In this survey paper, we aim to present and discuss the research papers related to medical document summarization published during the period, 2015 January to 2022 March. Our contributions are as follows: (1) We have provided a new categorization of different medical document summarization subtasks based on the type of medical document detailing each type with its individual challenges and datasets to provide insights about specific documents to natural language processing researchers, (2) We have also classified existing works based on input, output, and SS:DID NOT GET method used type to provide a NLP point of view , (3) We have also compiled different summary aspects that one must consider while evaluating medical summaries and then discussed new metrics developed in the community with their pros and cons.

 
 
 

## 3. Medical Document Summarization task

 
 In this section, we discuss different medical summarization tasks associated with different types of medical documents. Before diving into medical summarization, we broadly define the term automatic summarization. Automatic summarization can be defined as the task of computationally creating an abstract or a summary of original data while containing the relevant information and being consistent with the original data. Mathematically, automatic summarization is the task of producing the

 

 
 (1) | 
 | 
 Y s ​ u ​ m ​ m = f ⁡ ( D ) Y_{summ}=f(D) | 
 | 
 

 such that L ​ e ​ n ​ g ​ t ​ h ​ ( Y s ​ u ​ m ​ m ) L ​ e ​ n ​ g ​ t ​ h ​ ( D ) Length(Y_{summ}) Length(D) where Y s ​ u ​ m ​ m Y_{summ} is the output summary, D D is the input data and f ( . ) f(.) is the the summarization function. When this input data D D contains medical information, the task of producing the summary, Y s ​ u ​ m ​ m = f ⁡ ( D ) Y_{summ}=f(D) is known as medical summarization task.

 
 
 Inspired by ( Afantenos et al., 2005b ) , we can define medical information as the information and data concerning key concepts and techniques in the medical domain. Medical information may vary from core biomedical scientific concepts to a conversation between a doctor and a patient. Different types of medical information can be classified as follow:

 
 (1) 
 
 Biomedical Information: It concerns with all the core medical science, diseases, health, and nutrition theories and concepts.

 

 (2) 
 
 Clinical Information: It refers to all the information generated from an interaction between a medical institution and a patient comprising of the patient’s past medical records, medical reports generated from different tests, and the patient’s hospital admission history, etc.

 

 (3) 
 
 Conversational Information: It refers to all those semi-medical information that takes place on the web online or between doctor and patient such as medical queries and questions posted on different medical forums or a transcript of a conversation of a patient visiting a doctor.

 

 
 Based on these types of medical information and current literature, we can broadly divide the medical summarization task into different subtasks based on input data and documents associated with the medical domain (refer to Fig. 2 for distribution of existing works) as follows:

 
 • 
 
 Report Summarization: This refers to the summarization of notes or reports generated by a medical professional during the encounter with patients replacing input data D D in Equation 1 with clinical report. The main beneficiaries of summaries of the clinical reports are the medical practitioners as it will save their time and reduce their burden to go over the complete report.

 

 
 
 • 
 
 Health Record Summarization: A health record refers to all the medical documents generated during different stages of the "journey" after a patient’s admission to the hospital. Formally, Health Record Summarization can be defined as the multi-document summarization task where given a sequence of input documents D = { d } D=\{d\} , we have to generate a summary, Y s ​ u ​ m ​ m Y_{summ} . These health records are rich in medical information and summarizing these notes helps both doctor and patient to comprehend all the documents.

 

 
 
 • 
 
 Patient Health Question Summarization: Consumer Health Question (CHQ) refers to the questions asked by patients to professional doctors and medical experts. Therefore, CHQ Summarization can be defined as the task of summarization of these medical questions by replacing input data D D in Equation 1 with CHQ. The main SS: DID NOT LIKE THE TERM audience for consumer health question summarization is medical doctors and physicians. 

 

 
 
 • 
 
 Medical Dialogue Summarization: It can be defined as the process of summarizing a conversation over a digital platform or a physical encounter between a medical professional and a patient replacing the input data D D with Dialogue History. The users for Medical Dialogue Summarization are both patients and medical professionals as these summaries help to refer back to their interactions.

 

 
 
 • 
 
 Research Articles Summarization: The aim of biomedical summarization systems is to summarize and present important and relevant medical facts and information from long articles to medical professionals. The main target I HAVE OBJECTION WITH THIS WORD audience of this type of summarization are medical professionals and researchers.

 

 
 
 
 

## 4. Deep dive in medical tasks

 
 Before discussing the techniques and methods to solve MDS, we decided to first analyze and study the various types of MDS subtasks in detail. Since the MDS task is quite broad, it is important to categorize the works into different subtasks. For every subtask, we primarily discuss three issues: (1) Why this subtask is necessary, what type of document it deals with, and what is the structure of those documents?, (2) What are the specific challenges that each subtask poses? (3) What are publicly available datasets (refer to Fig 4 for distribution of datasets per subtask) for each subtask? We have also illustrated these
categorizations with a pictorial representation (Figure 3 ). We provided a comprehensive study of datasets in Table 1 .

 
 
 Figure 2. Illustration of distribution of existing work with respect to different MDS subtasks. Here, PQ: Patient Health Question, RA: Research Articles, HR: Health Records, MD: Medical Dialogue, RT: Report. 
 
 

### 4.1. Research Articles Summarization

 
 Medical research articles are the most dominant form of spreading and sharing medical research advancements. With the advent of internet, the number of research articles in biomedical domain has grown exponentially. For example, PubMed which is the most popular repository for medical research articles contains more than 32 million articles ( Moravvej et al., 2021 ) . Medical professionals face a lot of issues to keep themselves updated with the literature due to such a high rate of publications. This necessitates the need of automatic biomedical literature summarization systems which will help in reducing the burden of medical professionals. The aim of biomedical summarization systems is to summarize and present important and relevant medical facts and information from long articles to medical professionals. Most of the current works on research article summarization divide the articles on the following basis:

 
 (1) 
 
 Scientific medical research articles: These are all the general medical articles which are centered around medical science, diseases, medical theories and concepts.

 

 (2) 
 
 COVID-19 specific research articles: The emergence of the coronavirus has led to the explosion in the research articles discussing about the origin, symptoms, history and treatment of coronavirus. All the medical articles around COVID-19 come under this category.

 

 (3) 
 
 Health and Nutritional research articles: These types of research articles contain medical information and statistics about food, nutrition and health ( Shah et al., 2021c ) .

 

 (4) 
 
 Randomised Controlled Trials (RCTs): Randomised Controlled Trials can be defined as the experiments and trials that aim to study the efficacy and success of new medical treatments and interventions ( Hariton and Locascio, 2018 ) .

 

 
 We delineate the challenges, and dataset as follows:

 
 
 Challenges: The primary challenge in summarizing medical articles is to handle the length of these medical documents as they are usually very long and also can contain multiple documents. As medical articles are published on a daily basis, there is a need to continuously update the existing summaries with the advent of new articles while retaining the old information ( Park, 2020 ; Shah et al., 2021a ) . Shah et al. ( Shah et al., 2021c ) also highlighted the issue of unfaithful summaries due to the limitations of deep learning models to comprehend relations (such as negation) between different entities.
Other additional challenges include the low availability of some specific medical corpuses such as COVID-19 ( Kieuvongngam et al., 2020 ; Pasquali et al., 2021 ) and esoteric medical terminology that may not be present in generic datasets.

 
 
 Datasets: PubMed Open Access Subset 1 1 
 1 
 
 
 
 https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/ is an online repository of PubMed scholarly articles which contains millions of journal articles from PubMed. Wang et al. ( Wang et al., 2020a ) introduced a COVID-19 Open Research Dataset which includes 59,000 COVID-19 related research articles along with their corresponding summaries. Shah et al. ( Shah et al., 2021c ) proposed a high-quality health and nutritional dataset which consists of 7,750 scientific abstracts as the document and human written summaries by doctors of those abstracts as output summary. DeYoung et al. ( DeYoung et al., 2021 ) developed a multi-document biomedical scientific literature summarization dataset, MSˆ2 , which contains 470k documents and 20K summaries from biomedical literature. Wallace et al. ( Wallace et al., 2020 ) also introduced a dataset for summarization of Randomized Control Trials (RCTs) derived from the Cochrane platform 2 2 
 2 
 
 
 
 https://www.cochranelibrary.com/ . BIOASQ ( Tsatsaronis et al., 2015 ) is an open dataset containing 13 million PubMed research articles with their abstract as summaries of articles.

 
 
 

### 4.2. Health Record Summarization

 
 Electronic Health Records (EHR) ( Ambinder, 2005 ) are the digital documents that are used to document the encounter between medical professionals (physicians, doctors, nurses) and patients. EHR includes the complete course of the patient treatment from admission notes, doctor notes, nursing notes, and lab results to discharge notes. These records contain the complete the accounts of the journey of the patient in the hospital such as what happened to the patient, what treatment was given to the patient, and what are the future steps ( Adams et al., 2021 ) .
Summarizing these numerous notes helps both doctor and patient to comprehend all the documents as these health records are rich in medical information. Thus making it a necessity to automate the summary generation of these health records for the medical professional ( Black and Colford, 2017 ) . These summaries of EHR are commonly known as Discharge summaries ( Shing et al., 2021 ) . This task of generating discharge summaries can be viewed as a multi-document summarization task consisting of structured (such as lab results) and unstructured (such as nursing notes) documents. We delineate the challenges, and dataset as follows:

 
 
 Figure 3. Visual representation of MDS subtasks. 
 
 
 Challenges: The existing works ( Adams et al., 2021 ; Shing et al., 2021 ) on summarizing electronic health records highlight the following challenges: (1) Size of the input: As electronic health records contain a sequence of long documents, it can exceed the maximum memory limit of current models. This will lead to truncation and information loss, (2) Evidence: As current deep learning frameworks are black boxes ( Castelvecchi, 2016 ) , medical professionals need a way to trace the origin of information present in summaries back to original documents, (3) Faithfulness: Medical domain is a highly critical area where there is no buffer for hallucination. Hallucination can be defined as the information generated by the model which is not present in the input document ( Maynez et al., 2020a ) , (4) Hybrid nature of input: Adams et al. ( Adams et al., 2021 ) studied the EHR datasets and found that summaries in the dataset consist of both natures (Extraction and abstraction). This hybrid of extraction and abstraction warrants the use of models that can handle this transition between these two natures of summary, (5) Change in writing style: Adam et al. ( Adams et al., 2021 ) also noted that discharge summaries also include a substantial amount of re-writing the content of EHRs as there is a transition in writing summaries from health records as EHR usually contains information in chronological order but discharge summaries are written from the patient’s problems perspective, (6) Handling noisy reference summaries: Discharge summaries in EHR datasets usually either contains excessive information or a lack of critical medical information ( Adams et al., 2021 ) . Such noisy reference summaries can harm the model’s performance as the model will not be optimized for the right set of measures.

 
 
 Datasets: Shing et al. ( Shing et al., 2021 ) released a dataset of 6,000 encounters between patients and doctors. They derived the dataset from MIMIC-III clinical dataset ( Johnson et al., 2016 ) . MIMIC-III is a large open-source dataset that contains anonymous medical data of around 40,000 patients admitted to Beth Israel Deaconess Medical Center. Shing et al. ( Shing et al., 2021 ) only selected those encounters that contain admission notes, ICU notes, radiology notes, echo notes, ECG notes, and
discharge summaries. Adams et al. ( Adams et al., 2021 ) proposed a dataset called CLINSUM. CLINSUM contains medical records of 68,936 patients admitted to Columbia University Irving Medical Center from 2010 to 2014. They considered Brief Hospital Course (BHC) which is a mandatory section in discharge notes as the proxy reference summaries (Discharge summaries).

 
 
 

### 4.3. Report Summarization

 
 Clinical reports and notes convey information about detailed medical observations and findings of a medical encounter between a medical professional and a patient. Summarizing these reports is a very critical process as these summaries are the primary source of information while reviewing patient medical history and contain critical medical information ( Gershanik et al., 2011 ) . The most common type of clinical notes is radiology reports ( Hartung et al., 2020 ) . A radiology report is a medical document that contains the details of an imaging study (such as X-ray, MRI, etc). A Radiology report consists of three components: (1) Background section which contains the medical history of the patient, (2) Findings section which discusses the crucial observation and findings of the radiology study, and (3) Impression section which is a short summary of Findings section. Impression section is usually written by medical professionals which is a time-consuming process. The only aim of radiology report summarization is to automate the generation of this impression section. Most of the report summarization research revolves around the radiology domain because of the availability of large-scale datasets for radiology reports ( Demner-Fushman et al., 2015 ; Johnson et al., 2019 ) . However, all the existing work do not leverage the medical images in radiology reports to summarize the reports except for one work ( Delbrouck et al., 2021 ) .
We delineate the challenges, and dataset as follows:

 
 
 Challenges: Most of the current work in report summarization highlights the following two challenges: (1) Factual Correctness: Just like health record summarization, there is also no buffer hallucination and factual inconsistencies in report summarization, (2) Domain Specific Terminology: All the medical reports contain specific medical terminology that is usually not contained in normal/standard vocabulary and language models which warrants the use of external medical ontology and knowledge bases. Apart from this there is also need to understand the relationship between different medical terminology present in report.

 
 
 Datasets: MIMIC-CXR-JPG ( Johnson et al., 2019 ) is a large-scale freely available dataset of 377,110 chest x-rays associated with 227,827 imaging studies derived from the Beth Israel Deaconess Medical Center from 2011 to 2016. MIMIC-CXR-JPG is freely available on physionet 3 3 
 3 
 
 
 
 https://physionet.org/ which is an open repository of medical research data. Fushman et al. ( Demner-Fushman et al., 2015 ) released a dataset of 3,996 radiology reports from the Indiana Network for Patient Care and 8,121 associated images from the hospitals’ picture archiving systems. The images and reports were de-identified manually for ethical purposes.

 
 
 

### 4.4. Medical Dialogue Summarization

 
 Telemedicine has grown rapidly in the last few years with the aim of improving the efficiency of the healthcare system and reducing the workload of medical professionals. Telemedicine can be defined as the use of digital communication tools such as chatbots and chat interfaces with medical professionals to access the healthcare and medical services a person needs while practicing social distancing. With limited physical medical visits during the COVID-19 pandemic, telemedicine services and technologies have seen massive growth ( Mann et al., 2020 ) . Summarizing this conversation over telemedicine platforms is a significant step as it has multiple benefits: (1) Both doctor and patients have a record of their interaction, (2) the Doctor or patient can refer back to the conclusion or only to important parts of the conversation, and (3) as a means of passing information to other medical professionals ( Joshi et al., 2020 ) . The goal of medical dialogue summarization is to extract relevant medical facts, information, symptoms, and diagnosis. The generated summary can be either in the form of structured notes ( Krishna et al., 2021 ; Liu et al., 2019 ) or unstructured summaries. The most common type of medical notes are SOAP notes ( S ubjective information reported by patient, O bjective observations, A ssessment by medical professional and Future P lans) ( Krishna et al., 2021 ) . We delineate the challenges, and dataset as follows:

 
 
 Challenges: The main challenge that restricted the research in medical Dialogue Summarization as compared to text summarization in the past was the lack of publically released datasets. Another challenge in dialogue summarization that is not present in normal text summarization is the dynamic flow of information which means that relevant medical information and facts are scattered across the entire conversation between both speakers. An ideal medical dialogue summarization system must understand the conversation flow to connect scattered utterances as these conversations can be asynchronous. While overcoming these challenges, the medical dialogue summarization system should capture all the relevant and important medical facts and information from the conversation such as symptoms, diagnosis, etc.

 
 
 Datasets: Unlike normal text summarization, dialogue summarization is a much less explored area. But recently a lot of datasets have been developed to accelerate the research in medical dialogue summarization.
Krishna et al. ( Krishna et al., 2021 ) proposed a dataset of 6.5k clinical conversations along with SOAP notes as summaries. Song et al. ( Song et al., 2020a ) released a large dataset of 45k clinical conversations in the Chinese language. They map each conversation to two different summaries: (1) One summary of the medical problem reported by the patient, and (2) the Second summary of treatment suggested by a medical professional. Liu et al. ( Liu et al., 2019 ) proposed a structured summarization corpus of 100k dialogues where each dialogue is summarized into different predefined symptoms along with corresponding attributes. Joshi et al. ( Joshi et al., 2020 ) also created a corpus by extracting 25k medical conversations from a telemedicine platform and then hired doctors to annotate the conversations for corresponding summaries. Zhang et al ( Zhang et al., 2021 ) also build their own corpus of 1.3k medical conversations between doctors and patients.

 
 
 Figure 4. Illustration of dataset distribution with respect to different MDS subtasks. Here, PQ: Patient Health Question, RA: Research Articles, HR: Health Records, MD: Medical Dialogue, RT: Report. 
 
 
 

### 4.5. Patient health question summarization

 
 With the introduction of deep learning and transformer-based models, Question Answering (QA) systems has leveraged these techniques immensely. However, in the era of automation, there is still a dearth of works focused on one of the most relevant use cases, i.e., Medical QA. Recent surveys ( Jo et al., 2019 ; Rutten et al., 2019 ) also showed an increase in patients seeking medical information on the web these days. However, the main challenge impeding the research of Medical QA is the complexity of the questions posted by consumers/patients, both in the length of the question and the information contained in that question. These questions usually contain redundant and irrelevant information that is not required to answer the question by doctors or physicians. Roberts and Demner-Fushman (2016) highlighted this issue of irrelevant information in medical QA by showing that questions posted by patients are filled with more background information as compared to professional questions. Mrini et al. (2021c) also reports that patients often use medical terms and vocabulary different from

 
 
 Table 1. A study on datasets available for medical document summarization. ‘T’ stands for English text, ’SD’ stands for Single Document, ’MD’ stands for Multi-Document, ’SS’ stands for structured summary, ’MS’ stands for multiple summaries, ‘TC’ stands for Chinese text, ’TE’ stands for text (extractive) summary, ‘TA’ stands for text (abstractive) summary, ‘I’ stands for images. 
 
 
 
 ID Paper | 
 
 
 Type of Input data 
 | 
 
 
 Type of output data 
 | 
 
 
 Data Statistics 
 | 
 
 
 MDS subtask 
 | 

 
 #1: Wang et al. ( Wang et al., 2020a ) | 
 
 
 T, SD 
 | 
 
 
 TA 
 | 
 
 
 59,000 COVID-19 Research articles 
 | 
 
 
 Research Articles Summarization 
 | 

 
 #2: Shah et al. ( Shah et al., 2021c ) | 
 
 
 T, SD 
 | 
 
 
 TA 
 | 
 
 
 7,750 scientific articles 
 | 
 
 
 Research Articles Summarization 
 | 

 
 #3: DeYoung et al. ( DeYoung et al., 2021 ) | 
 
 
 T, MD 
 | 
 
 
 TA 
 | 
 
 
 470k documents
and 20K summaries from biomedical literature 
 | 
 
 
 Research Articles Summarization 
 | 

 
 #4: Tsatsaronis et al. (2015) | 
 
 
 T, SD 
 | 
 
 
 TA 
 | 
 
 
 13 million PubMed research articles 
 | 
 
 
 Research Articles Summarization 
 | 

 
 #5: Shing et al. ( Shing et al., 2021 ) | 
 
 
 T, MD 
 | 
 
 
 TA 
 | 
 
 
 dataset of 6,000 encounters between patients and doctors 
 | 
 
 
 Health Record Summarization 
 | 

 
 #6: Adams et al. ( Adams et al., 2021 ) | 
 
 
 T, MD 
 | 
 
 
 TA 
 | 
 
 
 medical records of 68,936 patients 
 | 
 
 
 Health Record Summarization 
 | 

 
 #7: Johnson et al. (2019) | 
 
 
 T, I 
 | 
 
 
 TA 
 | 
 
 
 Large scale freely available dataset of summaries of 377,110 chest x-ray s 
 | 
 
 
 Report Summarization 
 | 

 
 #8: Fushman et al. ( Demner-Fushman et al., 2015 ) | 
 
 
 T, I 
 | 
 
 
 TA 
 | 
 
 
 3,996 radiology reports from the
Indiana Network for Patient Care and 8,121 associated images from the hospitals’ picture archiving
systems 
 | 
 
 
 Report Summarization 
 | 

 
 #9: Krishna et al. ( Krishna et al., 2021 ) | 
 
 
 T,SD 
 | 
 
 
 TA, SS 
 | 
 
 
 dataset of 6.5k clinical conversations along with
SOAP notes as summaries 
 | 
 
 
 Medical Dialogue Summarization 
 | 

 
 #10: Song et al. ( Song et al., 2020a ) | 
 
 
 TC, SD 
 | 
 
 
 TE, MS 
 | 
 
 
 dataset of 45k clinical conversations 
 | 
 
 
 Medical Dialogue Summarization 
 | 

 
 #11: Liu et al. ( Liu et al., 2019 ) | 
 
 
 T, SD 
 | 
 
 
 TA, SS 
 | 
 
 
 structured summarization corpus of 100k dialogues 
 | 
 
 
 Medical Dialogue Summarization 
 | 

 
 #12: Joshi et al. ( Joshi et al., 2020 ) | 
 
 
 T, SD 
 | 
 
 
 TA 
 | 
 
 
 25k medical conversations from a
telemedicine platform 
 | 
 
 
 Medical Dialogue Summarization 
 | 

 
 #13: Zhang et al ( Zhang et al., 2021 ) | 
 
 
 T, SD 
 | 
 
 
 TA 
 | 
 
 
 1.3k medical conversations between doctors and patients 
 | 
 
 
 Medical Dialogue Summarization 
 | 

 
 #14: Abacha and Demner-Fushman (2019) | 
 
 
 T, SD 
 | 
 
 
 TA 
 | 
 
 
 1000 health questions along with their corresponding
manually written gold standard summaries 
 | 
 
 
 Patient health question summarization 
 | 

 
 #15: Mrini et al. (2021a) | 
 
 
 T, SD 
 | 
 
 
 TA 
 | 
 
 
 2,26,405 question summary pair from HealthCareMagic platform 
 | 
 
 
 Patient health question summarization 
 | 

 
 #16: Mrini et al. (2021a) | 
 
 
 T, SD 
 | 
 
 
 TA 
 | 
 
 
 31,062 question summary pairs from iCliniq. platform 
 | 
 
 
 Patient health question summarization 
 | 

 

 
 
 professional doctors. These issues necessitate the need for the summarization of consumer health questions (CHQ). Ben Abacha and Demner-Fushman (2019) also showed that summarization of CHQs

 
 
 improves the performance of QA systems by a margin of 58%. We delineate the challenges, and dataset as follows:

 
 
 Challenges: Like any other domain-specific summarization, one of the main challenges in CHQ summarization is capturing all the relevant medical domain terminologies and entities in the generated summary. A summarization system should also understand the relationship between different medical entities to make summaries more semantically consistent. Apart from these domain-specific challenges, there are also challenges that are distinct to the summarization of questions. The system should understand the type and focus of the question to make summary more meaningful. It should also capture all the sub-questions present in the original consumer question.

 
 
 Datasets: Recently, a few datasets have been proposed to aid the progress in CHQ summarization task. Abacha and Demner-Fushman (2019) proposed a dataset called MeQSum. MeQSum is a medical question summarization dataset that contains 1000 health questions along with their corresponding manually written gold standard summaries. Abacha and Demner-Fushman (2019) also showed two data augmentation techniques to increase the dataset size to 5,155 and 8,014 question summary pairs respectively. Mrini et al. (2021a) also created two CHQ datasets, extracting from a large-scale medical dialogue dataset MedDialog ( Zeng et al., 2020 ) named HealthCareMagic and iCliniq. HealthCareMagic contains a total of 2,26,405 question summary pairs whereas iCliniq contains 31,062 question summary pairs.

 
 
 
 

## 5. Organization of existing work

 
 There have been many attempts to solve the medical document summarization task, so it is important to categorize the existing works and methods to get a clear picture of the task and understand the current trends. We categorize the prior works into three broad categories, depending upon the variations in input, output, and techniques used. We have also illustrated these categorizations with a pictorial representation (Figure 5 ) and provided a comprehensive study in Table 3 (note that if some classifications are not marked in the table, then either the information about that category was not present, or is not applicable.).

 
 

### 5.1. On the basis of Input

 
 A summarization task is highly driven by the kind of input it is given. An existing
work can be distinguished from others on the basis of input in the following categories:

 
 
 Medical Domain Coverage: Depending upon the extent of medical domain coverage, we can classify existing works as medical domain-specific or medical generic. The approach to summarize a domain-specific input can differ from the generic
input greatly since feature extraction in the former can be very particular in nature and can have pre-requisites or specific standards to follow while generating summary. Most of the biomedical documents summarization ( Wallace et al., 2020 ; Guo et al., 2020 ; DeYoung et al., 2021 ; Moravvej et al., 2021 ; Du et al., 2020 ; Kedzie et al., 2018 ) are generic in nature since these articles contain information about almost all the medical
domains; whereas Covid-19 research articles summarization ( Kieuvongngam et al., 2020 ; Cai et al., 2022a ; Park, 2020 ) , Nutrition Articles summarization ( Shah et al., 2021a ) , Radiology report summarization ( Zhang et al., 2019b ; Hu et al., 2021 ) are some examples of medical domain-specific MDS tasks.

 
 
 Input Article Size: Since most of the work discussed in this survey has text modality as the input, the size of the text document in input can also be one way of categorizing all the related works. The summarization strategies might be different depending upon whether the textual input is a short single passage ( Abacha and Demner-Fushman, 2019 ; Savery et al., 2020 ; Mrini et al., 2021a ) or a multi-passage ( Zeng et al., 2020 ; Joshi et al., 2020 ; Delbrouck et al., 2021 ; Shing et al., 2021 ) . Patient health question summarization come under the single passage category and the rest of the MDS subtasks are multi-passage.

 
 
 Number of Input Documents: One way to categorize the existing works is by the number of text documents in the input i.e. Multi-document summarization that is the task of generating a summary that consists of information from multiple documents or single-document summarization that is the task of generating a summary from a single document. In MDS, multi-document summarization mainly includes Health Record Summarization ( Shing et al., 2021 ; Adams et al., 2021 ) and single-document summarization includes all other subtasks ( Abacha and Demner-Fushman, 2019 ; Savery et al., 2020 ; Mrini et al., 2021a ; Zeng et al., 2020 ; Joshi et al., 2020 ; Delbrouck et al., 2021 ) .

 
 
 Figure 5. Visual representation of proposed taxonomy based on input type, output type and method type. 
 
 
 

### 5.2. On the basis of Output

 
 We can group the existing works on the basis of type of output into the following categories:

 
 
 Kind of text summary: Most text summarization works fall into two categories: extractive and abstractive. Extractive summaries are those that contain information from the text directly, while abstractive summaries are those that involve creating a new summary that is focused on the overall meaning of the text. Depending on this, we can also classify MDS tasks as either extractive ( Du et al., 2020 ; Song et al., 2020b ) or abstractive ( Kieuvongngam et al., 2020 ; Cai et al., 2022a ; Park, 2020 ; Abacha and Demner-Fushman, 2019 ; Savery et al., 2020 ; Mrini et al., 2021a ; Zeng et al., 2020 ; Joshi et al., 2020 ; Delbrouck et al., 2021 ) .

 
 
 Structure of text summary: Structured text summary is a summary of a text that includes the most important information from the text in a well-organized format. An unstructured text summary is a summary of a text that includes some important information from the text but is not as well-organized as a structured text summary. On this basis, we can categorize current works into two categories: Structured MDS which generates a medical summary in a specific format such as SOAP notes ( Krishna et al., 2021 ) or symptoms with attributes ( Liu et al., 2019 ) and Unstructured MDS that generates medical summary in a normal generic paragraph fashion ( Shing et al., 2021 ; Du et al., 2020 ) .

 
 
 Table 2. Comprehensive list of works that uses different deep learning techniques to generate output summary. 
 
 
 
 
 
 Deep Learning Technique 
 | 
 
 
 Works using this technique 
 | 

 
 
 
 
 
 Seq2seq Learning Framework 
 | 
 
 
 Zhang et al. (2018) , Zhang et al. (2019b) , MacAvaney et al. (2019) , Sotudeh et al. (2020) , Shah et al. (2021b) , Moravvej et al. (2021) , Kedzie et al. (2018) , Song et al. (2020a) , Liu et al. (2019) 
 | 

 
 
 
 Transformer based Networks 
 | 
 
 
 Shing et al. (2021) , Kondadadi et al. (2021) , Kieuvongngam et al. (2020) , Park (2020) , Wallace et al. (2020) , Guo et al. (2020) , DeYoung et al. (2021) , Du et al. (2020) , Krishna et al. (2021) , Enarvi et al. (2020) , Chintagunta et al. (2021a) , Joshi et al. (2020) 
 | 

 
 
 
 Graph Deep Learning 
 | 
 
 
 Hu et al. (2021) , Cai et al. (2022a) 
 | 

 

 
 
 

### 5.3. On the basis of Method

 
 Different ways of solving the MDS problem have been developed, which can be divided into categories according to the method used as follows:

 
 
 Type of Summarization: Broadly, existing approaches to MDS can be classified into the following: Extractive MDS ( Du et al., 2020 ; Song et al., 2020b ) that involves the use of techniques that copies text from the input source itself; Abstractive MDS ( Kieuvongngam et al., 2020 ; Cai et al., 2022a ; Park, 2020 ; Abacha and Demner-Fushman, 2019 ; Savery et al., 2020 ; Mrini et al., 2021a ; Zeng et al., 2020 ; Joshi et al., 2020 ; Delbrouck et al., 2021 ) that involve rewriting the input document completely into a compressed form; and Hybrid (Extract-then-Abstract) ( Goel et al., 2021 ) MDS that first extracts salient and relevant information then paraphrases and compresses it to form final summary. The motivation for extract-then-abstract is that extractive models are better at being faithful to the source, but abstractive models are better at producing coherent summaries; thus combining the best from both worlds to produce a faithful and fluent summary.

 
 
 Table 3. Comprehensive study of existing work using the proposed taxonomy (refer to Section 5 ) . 
 
 
 
 
 | 
 Input Based | 
 Output Based | 
 Method Based | 

 
      Papers | 
 MDC | 
 IAS | 
 IDN | 
 KTS | 
 STS | 
 TS | 
 LP | 
 NI | 
 KB | 

 
 
 
 Specific Medical Domain . 

 | 
 
 
 Generic . 

 | 
 
 
 Single-Passage . 

 | 
 
 
 Multi-Passage . 

 | 
 
 
 Single-doc . 

 | 
 
 
 Multi-doc . 

 | 
 
 
 Extractive . 

 | 
 
 
 Abstractive . 

 | 
 
 
 Structured summary . 

 | 
 
 
 Unstructured summary . 

 | 
 
 
 Abstractive . 

 | 
 
 
 Extractiive . 

 | 
 
 
 Hybrid . 

 | 
 
 
 Rule Based . 

 | 
 
 
 Machine learning . 

 | 
 
 
 Deep learning . 

 | 
 
 
 Consistency . 

 | 
 
 
 Copy mechanism . 

 | 
 
 
 Other . 

 | 
 
 
 KB used . 

 | 
 
 
 KB not used . 

 | 

 
 Shing et al. (2021) | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 Kieuvongngam et al. (2020) | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 Sarkar et al. (2011) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 Cai et al. (2022a) | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 

 
 Park (2020) | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 Wallace et al. (2020) | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 

 
 Shah et al. (2021c) | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 Shah et al. (2021b) | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 ✓ | 

 
 Guo et al. (2020) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 DeYoung et al. (2021) | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 Moravvej et al. (2021) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 Du et al. (2020) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 

 
 Kedzie et al. (2018) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 

 
 Sarkar (2009) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 

 
 Krishna et al. (2021) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 

 
 Enarvi et al. (2020) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 

 
 Chintagunta et al. (2021a) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 Song et al. (2020a) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 

 
 Liu et al. (2019) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 

 
 Joshi et al. (2020) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 

 
 Zhang et al. (2021) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 Zhang et al. (2018) | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 

 
 Aramaki et al. (2009) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 MacAvaney et al. (2019) | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 

 
 Zhang et al. (2019b) | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 ✓ | 

 
 Gayathri and Jaisankar (2015) | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 Hu et al. (2021) | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 

 
 Kondadadi et al. (2021) | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 Mrini et al. (2021b) | 
 | 
 ✓ | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 Abacha and Demner-Fushman (2019) | 
 | 
 ✓ | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 

 
 Yadav et al. (2022) | 
 | 
 ✓ | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 | 
 ✓ | 
 ✓ | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 ✓ | 
 ✓ | 
 | 

 

 
 
 
 Learning process: Most of the existing techniques to solve the task of MDS can fall under these three categories: (1) Rule-based learning process ( Sarkar, 2009 ; Gayathri and Jaisankar, 2015 ) : This involves the use of sentence ranking, sentence extraction, and clustering techniques. Most of these works are pre-deep learning era (before 2010), (2) Machine learning based ( Aramaki et al., 2009 ; Sarkar et al., 2011 ) : This involves the use of classical machine learning techniques such as supervised clustering techniques, and (3) Deep learning ( Kieuvongngam et al., 2020 ; Cai et al., 2022a ; Park, 2020 ; Abacha and Demner-Fushman, 2019 ; Savery et al., 2020 ; Mrini et al., 2021a ; Zeng et al., 2020 ; Joshi et al., 2020 ; Delbrouck et al., 2021 ) : With the advent of neural networks and transformers, most of the current MDS approaches leverage these techniques. Table 2 lists the works using different types of deep learning techniques.

 
 
 Notion of importance: Unlike other domains, medical domain is a very high stake area that requires certain summarization models to give more focus or weightage to certain aspects of the summary. Thus, the most significant distinction between the existing work would be the notion of importance used to generate the final summary. A diverse set of objectives ranging from consistency ( Zhang et al., 2019b ) , copy mechanism ( Zhang et al., 2018 ) , and more weightage to extraction ( Goel et al., 2021 ) have been explored in an attempt to solve the MDS task in an efficient manner.

 
 
 Use of Knowledge Base: External Knowledge bases (KBs) can improve natural language processing (NLP) in a number of ways ( Zouhar et al., 2022 ) especially in the medical domain as medical information uses very specific terminologies and vocabulary than general information. Based on this, we can classify current MDS works on the basis of whether they use a KB ( MacAvaney et al., 2019 ; Sotudeh et al., 2020 ; Hu et al., 2021 ; Cai et al., 2022a ) or not ( Shing et al., 2021 ; Adams et al., 2021 ; Delbrouck et al., 2021 ; Zhang et al., 2019b ) . If they use KB, what type of KB they are using is crucial as there can be multiple ways in which a KB can be used to improve performance such as using a medical database directly as an external knowledge source ( MacAvaney et al., 2019 ; Sotudeh et al., 2020 ) or creating a knowledge graph from a dataset to create a local knowledge base ( Hu et al., 2021 ; Cai et al., 2022a ) . A detailed list of works that use knowledge bases networks can be found
at Table 4 .

 
 
 Table 4. Comprehensive list of works that use specific Knowledge bases to generate output summary. 
 
 
 
 
 
 Knowledge Bases (KB) 
 | 
 
 
 Works using this KB 
 | 

 
 
 
 
 
 Pre-trained medical Transformer as base model 
 | 
 
 
 Wallace et al. (2020) , Du et al. (2020) , Yadav et al. (2022) 
 | 

 
 
 
 Medical Database such as QuickUMLS ( Soldaini, 2016 ) 
 | 
 
 
 Zhang et al. (2018) , MacAvaney et al. (2019) 
 | 

 
 
 
 Knowledge Graphs 
 | 
 
 
 Hu et al. (2021) , Cai et al. (2022a) 
 | 

 

 
 
 
 

## 6. Evaluation Techniques

 
 There have been a lot of diverse medical input documents and numerous attempts in solving the medical task summarization task. However, the evaluation of these summaries is still an open problem. Most of the existing works use standard n-gram matching metrics, (Recall-Oriented Understudy for Gisting Evaluation) ROUGE ( Lin, 2004 ) , Metric for Evaluation of Translation with Explicit ORdering (METEOR) ( Banerjee and Lavie, 2005 ) , and Bilingual Evaluation Understudy (BLEU) ( Papineni et al., 2002 ) as evaluation measures to evaluate the quality of generated summaries. A few works also uses semantic evaluation metrics such as Sentence Mover Similarity (SMS) ( Clark et al., 2019 ) and BERTScore ( Zhang et al., 2019a ) . The most popular way of measuring the quality of text summarization in the field is the Recall-Oriented Understudy for Gisting Evaluation (ROUGE), which is largely reliant on the similarity between the generated summary and the reference summary in terms of n-gram overlap. However, a major downside to ROUGE is its need for a reference summary of good quality. This was highlighted in many studies ( Fabbri et al., 2020 ; Jain et al., 2022 ) , that pointed out that some reference summaries are not of consistent and satisfactory quality.

 
 
 However, the best way to evaluate the quality of a generated summary is to perform extensive human evaluations. The standard predefined metrics ( Cai et al., 2022b ; Wallace et al., 2020 ; Shah et al., 2021a ) to measure in the human evaluation process are (1) Informativeness or Relevance: It measures how many important and relevant facts and information summary are able to retain, (2) Coherence: measures that summary sentences or paragraphs have a smooth logical transition, (3) Redundancy: It measures that summary should not contain repeated information or facts, (4) Fluency: It measures the grammatical correctness of summary, (5) Consistency or Factuality: It measures the factual correctness of summary with respect to the source article, and (6) Contradiction: It measures whether there is some information present in the summaries that contradict some other information or at disagreement with another piece of information. Moramarco et al. ( Moramarco et al., 2021 ) conduct a human evaluation of the quality of Clinical SOAP notes generated by state-of-the-art (SOTA) text summarization models (Both extractive and abstractive models.) by asking evaluators to count medical facts, and computing precision, recall, F-score, and accuracy from those raw counts. One fascinating observation from this study was that both abstractive and extractive models don’t hallucinate any medical facts as abstractive models also tend to copy many phrases from the source text. However, they observed that the only hallucination that the abstractive models do in those samples is a numerical one showing the limitations of current SOTA models to comprehend numerical relations. This limited numerical literacy of SOTA language models is also highlighted in many previous studies ( Thawani et al., 2021b ) . However, human evaluations are very expensive and time-consuming, thus not scalable for large models and datasets.

 
 
 Figure 6. Visual representation of specific attributes to be considered while doing evaluation of medical summaries 
 
 
 There are certain attributes that are specific to medical documents which above mentioned metrics fail to measure (Figure 6 ). Following are the unique traits that one must consider while dealing with medical documents: (1) Comprehensive summaries: Adams et al. ( Shing et al., 2021 ) study that medical summaries are densely packed with medical terminologies and entities. They found out that 20.9% of
the words in a summary is medical entities as per UMLS ( Bodenreider, 2004 ) dataset as compared to 14.1% in the source document. They also showed that summaries contain 26 unique medical entities on average whereas source document contains 265 unique medical entities. This is a compression ratio of only 10 as compared to the compression ratio of 45 for all tokens. Goel et al. ( Adams et al., 2021 ) highlight that performance degrades when the number of entities increases in the CNN/DailyMail dataset. This necessitates the need for domain-specific fact-based evaluation approaches which can encode a deeper knowledge of clinical concepts and their complex semantic and temporal relations to assess the quality of generated medical entities. (2) Hallucination: Hallucination is one of the leading limitations of existing summarization systems. In the case of clinical and medical settings, hallucinations in summaries can lead to serious treatment misjudgment and errors. Wallace et al. ( Wallace et al., 2020 ) also highlighted that metrics like ROUGE fail to capture hallucinations in summaries. Some studies ( Shah et al., 2021c ; Zhang et al., 2021 ) also showed this issue of hallucinated content and medically incorrect entities and outputs in generated summaries through human evaluation. (3) Faithfulness: Shing et al. ( Goel et al., 2021 ) define faithful summary as a summary that is faithful to the source (doesn’t contain any information from outside of the source and whatever present should be factually correct) and relevant as measured by the reference summary (contains all the key and salient information). (4) Content Organization: Content organization refers to the structure of a document and the best metric to measure this is Coherence ( Fabbri et al., 2020 ) . Barzilay et al. ( Barzilay and Lapata, 2005 ) define coherence as information about the same entities are considered to be more coherent than information with the abrupt transition of topics or entities . However, Adams et al. ( Shing et al., 2021 ) showed that CLINSUM (dataset of health record summarization) has a swift and abrupt topic and entity transitions with only 34% of the entities repeating. They also showed that ROUGE is insufficient to capture the coherence in the CLINSUM dataset using a pairwise ranking approach ( Barzilay and Lapata, 2005 ) . This problem warrants the need of developing domain-specific models of coherence, which can handle these abrupt topic shifts, and able to capture these relations between different medical entities present in the summaries. Some studies ( Fabbri et al., 2020 ; Fabbri et al., 2020 ; Bhandari et al., 2020 ) also showed the importance and need of domain-oriented evaluation metrics. (5) Noisy Reference Summaries: Dependency on a good quality reference summary is one of the major limitations of metrics like ROUGE or BLEU ( Kryscinski et al., 2019 ; Fabbri et al., 2020 ) as reference summary itself can be of below-par quality. Kripalani et al. ( Kripalani et al., 2007 ) discovers that discharge summaries often miss important and salient information such as diagnostic results (missing from 33%-
63% time), treatment course of a patient(7%-22%), discharge medications (2%-40%), test
results (65%), counseling (90%-92%), and follow-up and future steps (2%-43%). This shows that complete dependency on reference summary for evaluation is not ideal in clinical settings as reference summary in itself lacks critical information.

 
 
 From the above discussion, it is evident that we need new evaluation metrics that are capable of measuring all these aspects of a summary of a medical document. Recently various attempts have been made by the community to propose such efficient and domain-specific metrics. Some of the contributions are as follow: (1) KG metrics ( Shah et al., 2021a ) : Shah et al. ( Shah et al., 2021a ) propose two metrics KG(G) and KG(I) to capture relevance and faithfulness of generated summary respectively. Both these metrics are based on entity and relation matching.
KG(G) captures relevance by calculating the number of overlapping entity-relation-entity pairs in the generated summary and reference summary. Similarly, KG(I) calculates the number of overlapping entity-relation-entity pairs in the generated summary and input source document to measure the faithfulness of the summary with respect to the input document. To calculate the entity-relation-entity pairs for generated summaries, gold summaries, and input documents, they run an entity tagging model and relation classification model on all these three documents. Then they match the gold summaries
 ( e i G , e j G , r G ) (e^{G}_{i},e^{G}_{j},r^{G}) pairs and input document ( e i I , e j I , r I ) (e^{I}_{i},e^{I}_{j},r^{I}) pairs using cosine similarity with the generated output summary ( e i o , e j o , r o ) (e^{o}_{i},e^{o}_{j},r^{o}) pairs. They consider all those pairs to be a match that has a cosine score of more than a fixed threshold of 0.7. (2) Aggregation Cognisance (Ag) metric: Shah et al. ( Shah et al., 2021a ) proposes a metric Aggregation Cognisance (Ag) to capture the capability of the model to generate output summary that is aware of the right aggregation or entailment (contradiction or agreement) from the input source document by using a classifier to measure and compare the entailment in the generated output summary and the input source document. (3) Diversity metric ( Rao and Daumé, 2019 ; Shah et al., 2021a ) : Rao et al. ( Rao and Daumé, 2019 ) proposed a metric to measure diversity in model’s outputs by computing the ratio of unique trigrams present in a generated summary. (4) Readability evaluation metrics ( Guo et al., 2020 ) : Guo et al. uses Flesch-Kincaid grade level ( Kincaid et al., 1975 ) , Gunning fog index ( Flory et al., 1992 ) ,
and Coleman-Liau index ( Coleman and Liau, 1975 ) to compute the ease of readability and fluency of generated summaries. (5) Fact-based Evaluation: To measure the accuracy of medical facts generated by the model in output summary, Enarvi et al. ( Enarvi et al., 2020 ) utilizes a machine-learning clinical fact
extractor module that is capable of extracting medical facts such as treatment or diagnosis and fine-grained attributes such as body part or medications. They use this model to extract facts from the generated summary and reference summary and calculate the F1 score for both extracted facts. Similar type of factual correctness metrics has been used in several other works ( Hu et al., 2021 ; Zhang et al., 2019b ; Chintagunta et al., 2021b ) . Enarvi et al. ( Enarvi et al., 2020 ) used Negex metric ( Harkema et al., 2009 ) to capture the capability of model to evaluate the negative status of medical concepts by computing the negations in generated summary and calculating whether the negations were accurate for the medical facts present in the generated summary. (6) Faithfulness and Hallucination Metrics: Shing et al. ( Shing et al., 2021 ) proposes two evaluation metrics, Faithfulness-adjusted Precision and Incorrect Hallucination Rate to measure faithfulness and hallucination in generated summaries. Faithfulness-adjusted Precision is defined as:

 

 
 (2) | 
 | 
 F ​ a ​ i ​ t ​ h ​ f ​ u ​ l ​ n ​ e ​ s ​ s ​ a ​ d ​ j ​ u ​ s ​ t ​ e ​ d ​ P ​ r ​ e ​ c ​ i ​ s ​ i ​ o ​ n = ( O ∩ G ∩ S ) O FaithfulnessadjustedPrecision=\frac{(O\cap\ G\cap S)}{O} | 
 | 
 

 where O O refers to generated output summary, G G refers to gold reference summary, and S S refers to input source document. In a similar fashion, Incorrect Hallucination Rate is calculated as:

 

 
 (3) | 
 | 
 I ​ n ​ c ​ o ​ r ​ r ​ e ​ c ​ t ​ H ​ a ​ l ​ l ​ u ​ c ​ i ​ n ​ a ​ t ​ i ​ o ​ n ​ R ​ a ​ t ​ e = ( O − ( O ∩ G ) − ( O ∩ S ) ) O IncorrectHallucinationRate=\frac{(O-(O\cap G)-(O\cap S))}{O} | 
 | 
 

 
 
 Overall, the discussed metrics have still their limitations; however, there is a great scope for future improvement in the area of evaluation techniques for medical summaries. We provided a comprehensive study of these evaluation metrics with their pros and cons in Table 5 .

 
 
 

## 7. Ehtical Consideration

 
 Deep learning is a powerful tool that can be used for good or bad. On the one hand, deep learning can be used to create amazing new technology that can be used to improve our lives. On the other hand, deep learning can also be used for malicious purposes, such as creating deepfakes or propaganda bots. As with any technology, there are ethical considerations to be taken into account when using deep learning. Some of the ethical considerations of deep learning include data privacy, data bias, and the impact of artificial intelligence on society. The ethical implications of clinical deep learning are still being explored.
However, there are some potential concerns that have been raised.

 
 
 Table 5. Comparative study of evaluation techniques for Medical Document Summarization. 
 
 
 
 
 
 Metric name corresponding paper 
 | 
 
 
 Pros Cons 
 | 

 
 
 
 
 
 KG metrics. Shah et al. (2021a) 
 | 
 
 
 Advantages 
 | 

 
 | 
 
 
 - Designed to capture relevance and faithfulness of generated summary with respect to reference summary and the input source document. 
 | 

 
 | 
 
 
 Disadvantages 
 | 

 
 | 
 
 
 - dependent on a good quality reference summary and Performance may vary with the use of different entity tagging and relation classification models 
 | 

 
 
 
 Aggregation Cognisance (Ag) metric. Shah et al. ( Shah et al., 2021a ) 
 | 
 
 
 Advantages 
 | 

 
 | 
 
 
 - Designed to capture the capability of a model to generate an output summary that is aware of the right
entailment (contradiction or agreement) with respect to input source. 
 | 

 
 | 
 
 
 Disadvantages 
 | 

 
 | 
 
 
 - Performance may vary with the use of different entailment classifiers. 
 | 

 
 
 
 Diversity metric. Rao and Daumé (2019) ; Shah et al. (2021a) 
 | 
 
 
 Advantages 
 | 

 
 | 
 
 
 - Aims to Capture the diversity in generated output summary 
 | 

 
 | 
 
 
 Disadvantages 
 | 

 
 | 
 
 
 - The technique is very naive as it considers only n-gram overlaps. 
 | 

 
 
 
 Readability evaluation metrics. Guo et al. (2020) 
 | 
 
 
 Advantages 
 | 

 
 | 
 
 
 - Designed to measure the readability and fluency of generated output. 
 | 

 
 | 
 
 
 Disadvantages 
 | 

 
 | 
 
 
 - The technique may not be robust as it uses rule-based metrics. 
 | 

 
 
 
 Fact-based Evaluation. Enarvi et al. ( Enarvi et al., 2020 ) 
 | 
 
 
 Advantages 
 | 

 
 | 
 
 
 - Aims to measure the factual correctness of medical facts present in generated summary. 
 | 

 
 | 
 
 
 Disadvantages 
 | 

 
 | 
 
 
 - Different works use different fact extractor leading to a lot of inconsistencies among results. 
 | 

 
 
 
 Faithfulness and Hallucination Metrics. Shing et al. ( Shing et al., 2021 ) 
 | 
 
 
 Advantages 
 | 

 
 | 
 
 
 - Aims to measure faithfulness of summaries and detection of hallucination in summaries. 
 | 

 
 | 
 
 
 Disadvantages 
 | 

 
 | 
 
 
 - The technique is very naive as it considers only syntactic overlap. 
 | 

 

 
 
 Bias and Fairness: Machine learning is a data-driven field that makes the models and algorithms susceptible to the social bias present in the datasets. This will make the models to make decisions that are skewed against certain communities of the society ( Garrido-Muñoz et al., 2021 ) . Chen et al. ( Chen et al., 2019 ) conducted two studies to understand the bias in clinical deep learning models, one for intensive care unit (ICU) mortality prediction and second for 30-day psychiatric readmission prediction. They reported that these models underperform on both tasks with respect to
women, ethnic and racial minorities. Zhang et al. ( Zhang et al., 2020a ) also studied the societal bias in language models and found that they too are susceptible to biases by recommending hospitals and clinics to white people and prison to black people which is very worrisome sign and raises a question mark on real world application of such systems. This warrants the need of debiased and fair machine learning techniques ( Correa et al., 2021 ; Luo et al., 2022 ; Yang et al., 2022 ) . Similar types of studies about bais and debiasing techniques must be applied to medical summarization systems too before deploying them in real-world settings.

 
 
 Data Privacy: The digitisation of medical records is underway for the convenience of patients, doctors, and medical staff. The digitization of medical documents is a key part of medical care and is also of great importance in improving medical care. The digitization of medical documents has many benefits. It helps to save time and paper and to improve the quality of medical care. It also helps to reduce medical errors. The digitization of medical documents is also of great importance in terms of both medical research and clinical process. The digitization of information has led to a number of privacy concerns. The main concern is that digitization makes it easier for organizations to collect, store and use personal data which makes the patient’s data susceptible to various security attacks
 ( Priya et al., 2017 ; Kaye, 2012 ) . This requires the need for data anonymization which is the process of de-identifying data so that it can no longer be traced back to an individual; this can be done by removing or encrypting personal identifiers such as names, addresses, and social security numbers ( Olatunji et al., 2022 ) . Data anonymization is a critical process that must be done before creating or releasing any dataset for the MDS task.

 
 
 

## 8. Discussion

 
 Medical Document summarization has been researched quite frequently in recent years with multiple diverse models. We discuss how medical documents can be categorized into fine-grained categories with each type presenting its unique challenges to deal with. It can be seen that some documents (Research articles, Radiology reports, and medical dialogue) have been explored more than other medical documents (Electronic health records and consumer health questions). The reason for this can be attributed to the difficulty in obtaining the datasets for Electronic health records and Consumer Health Questions sub-tasks. We also categorize the techniques to solve Medical Document summarization to understand the current trends.
We also categorized current works on the basis of input, output, and method used. It is evident that most of the current work revolves around single documents as input, abstractive summaries as outputs, and deep learning or transformer as their base model. Some of the works also use external Knowledge bases in the form of medical databases or knowledge graphs to further improve the performance. There are also few that focus on specific medical domains such as COVID articles or Radiology reports. Recently, the community is also moving towards a hybrid of extractive and abstractive summarization approaches (Extract-then-Abstract) to improve the faithfulness of generated summaries.
After conducting a comprehensive study of evaluation metrics, we can make the following observations: (1) Standard evaluation metrics are insufficient to capture the special aspects of medical summaries. (2) There are a lot of inconsistencies in the human evaluation process as different researchers measure different aspects of the summary; thus we compile and report all these aspects that need to be considered while doing the human evaluation. (3) We also study the existing medical domain-specific metrics that the community should use alongside standard evaluation metrics. In addition to all these, we also discuss the need of ensuring that deep learning is used responsibly and with consideration for the potential impacts on individuals and society. Broadly, there are two ethical concerns that one must consider while deploying their model in the real world; Fairness: one must ensure that their algorithm is used fairly and free from bias, and data privacy: one must ensure that the data collected is kept secure and that unauthorized access is not granted.

 
 
 

## 9. Future Work

 
 The MDS task is relatively new, and the work done so far has only scratched the surface of what
this field has to offer. In this section, we discuss the future scope of the MDS task, including some
possible improvements in existing works, as well as some possible new directions.

 
 

### 9.1. Scope of improvement

 
 Better evaluation metrics: Most of the existing works still use standard evaluation techniques like ROUGE scores and BLEU scores to measure the quality of the generated medical summary. However, these metrics suffer from a lot of limitations. Many works try to compensate for the limitations of these metrics by conducting a human evaluation procedure. But these are very expensive and time-consuming processes making them unscalable for larger datasets. Many attempts have been made to formulate evaluation metrics that are capable of measuring medical summaries specific aspects such as readability, faithfulness, fact-based evaluation, etc. However, all these metrics suffer from their own limitations making a room for the scope of improvement which are discussed as follows: (1) All of these metrics work at a syntactic level which makes them unsuitable to capture semantic overlaps. (2) Most of these metrics are over-dependent on reference summary. (3) There are no metrics that are designed to work in multimodal settings. (4) There is also a need for metrics that are capable of measuring the correctness of the relationship between different generated medical concepts by designing medical domain-aware metrics. We need to overcome all these limitations in order to improve overall medical summarization systems.

 
 
 Multimodality: Generating multi-modal summaries in the medical domain is an unsolved problem. With medical history summarization systems, generating relevant text summaries accompanied by relevant sonograms, cardiograms, or other relevant visual/aural content is an intuitive next step. There have been several unsupervised efforts to generate multi-modal summaries ( Jangra et al., 2020a ; Jangra et al., 2020b ; Jangra et al., 2021b ) ; however, the community still lacks relevant datasets and models to obtain supervised multi-modal summarization systems for the medical domain.

 
 
 Faithful summaries: One open challenge in the current summarization system is the unfaithfulness of generated summaries with respect to the input document ( Chen et al., 2022 ; Maynez et al., 2020b ) . Solving this problem of unfaithful and inconsistent summaries becomes of utmost priority when working in the medical domain as a minor error can lead to a wrong judgment or decision. Due to this reason, the community either depends on extractive summarization techniques or uses the extract-then-abstract ( Shing et al., 2021 ) methodology to create an evidence fallback system to check the validity of generated summary. Hence the community still lacks an abstractive summarization model that is capable of generating faithful and factually consistent summaries.

 
 
 

### 9.2. New directions

 
 Focus on Numerical Literacy: Moramarco et al. ( Moramarco et al., 2021 ) showed that abstractive models tend to numerical errors while generating summaries showing the limitations of current SOTA models to comprehend numerical values and relations. Thawani et al. ( Thawani et al., 2021a ; Thawani et al., 2021b ) also discussed how numbers are often neglected in natural language processing and thus many SOTA language models have limited numerical
literacy. Numerical values are one of the salient features of medical documents such as patient’s body vital information or lab results ( Bigeard et al., 2015 ) . Thus developing summarization systems that are capable of numerical reasoning is of primary importance.

 
 
 GPT-3 based data annotation: Obtaining datasets of medical documents is one of the biggest challenge faced by the community restricting the researchers to only a few datasets available online. The two major reasons for this are the privacy of patient information and if the documents are anonymized, expensive manual annotation of those documents as this can be done only by medical professionals. Chintagunta et al. ( Chintagunta et al., 2021a ) showed how GPT-3 ( Brown et al., 2020 ) can be used as a tool to generate a dataset for medical dialogue summarization task. They reported that they can obtain the same performance as that of a human-labeled dataset with a 30x smaller amount of human-labeled. GPT-3 is a few-shot learner language model; thus showing only a few human-labeled examples, it can be used to annotate or label the rest of the data. Similar strategy can be used to develop datasets for related MDS subtasks.

 
 
 Explainable summarization: With the introduction of explainable artificial intelligence (AI) ( Arrieta et al., 2020 ) , it is required
to provide explanations/interpretations behind any decision taken
by a machine learning algorithm especially in the medical domain which is a high stake domain. Medical professionals need an interpretable mechanism to understand why the model generated that particular information. However, most of the existing interpretable work revolves around classification tasks ( Ribeiro et al., 2016 ; Sundararajan et al., 2017 ) which can not be used to interpret natural language generation models. Xu et al. ( Xu and Durrett, 2021 ) propose an ablation and attribution strategy to interpret a summarizer’s decision at every time step. however, there is no existing research on explainable summarization in the medical domain showing a research gap between both these areas which needs to be filled.

 
 
 Query-based summarization: A lot of work has been done in query-based text summarization ( Litvak and Vanetik, 2017 ; Rahman and Borah, 2019 ) which aims at extracting information that tries to answer a query about input source text. This type of use case can be very relevant to medical domain as medical professionals can leverage this feature of query based summarization by generating only relevant summary related to medical professional or patient query from
plethora of documents. We believe that query-based summarization setup can imporve the overall adaption of MDS tasks in real-world settings.

 
 
 Adversarial study of medical summarization systems: Recent works have showed that adversarial examples can be generated by applying small
perturbations or noise to the inputs which will make deep learning models to perfrom abnormally ( Huang et al., 2011 ) . The robustness of medical summarization systems for medical documents such as electronic health records is especially critical because of the high public interests of such a high stake domain that make it more susceptible to such adversarial attacks. Wang et al. ( Wang et al., 2020b ) showed the vulnerability of the existing summarization systems by constructing adversarial samples for EHR data. They also proposed a novel defense strategy to detect adversarial examples in datasets. In order to deploy such medical summarization systems, this type of testing needs to be done to test the robustness of models.

 
 
 Fair and ethical summarization systems: Despite so much research being done around Fair and Ethical NLP ( Fort and Couillault, 2016 ) , there is still no work discussing the fairness of medical summarization systems. Fair and ethical summarization systems are those systems that are developed with the goal of producing summary outputs that accurately and ethically reflect the original source material. These systems must be designed to ensure that all potentially sensitive or proprietary information is kept secure, and that summary outputs are not biased toward any particular point of view.

 
 
 
 

## 10. Conclusion

 
 The internet has drastically changed the way medical documents are created and accessed. In the past, they were often handwritten which made them hard to share and find. Now, they are typically created electronically which makes them much easier to both access and share. The internet has also allowed medical professionals to easily share documents with one another which has ultimately improved patient care and medical research. This paper provides a survey to introduce users and researchers to the techniques and current trends in the Medical Summarization task. We cover the formal definition of the Medical Summarization task, a detailed analysis of different medical tasks based on the type of medical documents and specific datasets and challenges associated with them, a detailed categorization of existing works based on input, output, and technique, and an in-depth look at the evaluation metrics utilized to measure the quality of the summaries. To finish, we suggest some potential future directions for further research. We are confident that this survey will encourage more work in medical document summarization.

 
 
 

## References

 
 
 Abacha and Demner-Fushman (2019) 
 
Asma Ben Abacha and Dina
Demner-Fushman. 2019.

 
 On the Summarization of Consumer Health Questions.
In ACL .

 
 
 

 
 Adams et al . (2021) 
 
Griffin Adams, Emily
Alsentzer, Mert Ketenci, Jason Zucker,
and Noémie Elhadad. 2021.

 
 What’s in a Summary? Laying the Groundwork for
Advances in Hospital-Course Summarization.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2105.00816 

 

 
 Afantenos et al . (2005a) 
 
Stergos Afantenos,
Vangelis Karkaletsis, and Panagiotis
Stamatopoulos. 2005a.

 
 Summarization from medical documents: a survey.

 
 Artificial intelligence in medicine 
33, 2 (2005),
157–177.

 
 
 

 
 Afantenos et al . (2005b) 
 
Stergos D. Afantenos,
Vangelis Karkaletsis, and Panagiotis
Stamatopoulos. 2005b.

 
 Summarization from Medical Documents: A Survey.

 
 Artificial intelligence in medicine 
33 2 (2005), 157–77.

 
 
 

 
 Ambinder (2005) 
 
Edward P Ambinder.
2005.

 
 Electronic health records.

 
 Journal of oncology practice 
1, 2 (2005),
57.

 
 
 

 
 Apostolidis et al . (2021) 
 
Evlampios Apostolidis,
Eleni Adamantidou, Alexandros I Metsai,
Vasileios Mezaris, and Ioannis Patras.
2021.

 
 Video summarization using deep neural networks: A
survey.

 
 Proc. IEEE 109,
11 (2021), 1838–1863.

 
 
 

 
 Aramaki et al . (2009) 
 
Eiji Aramaki, Yasuhide
Miura, Masatsugu Tonoike, Tomoko Ohkuma,
Hiroshi Mashuichi, and Kazuhiko Ohe.
2009.

 
 TEXT2TABLE: Medical Text Summarization System
Based on Named Entity Recognition and Modality Identification. In
 Proceedings of the BioNLP 2009 Workshop .
Association for Computational Linguistics,
Boulder, Colorado, 185–192.

 
 
 https://aclanthology.org/W09-1324 

 

 
 Arrieta et al . (2020) 
 
Alejandro Barredo Arrieta,
Natalia Díaz-Rodríguez, Javier
Del Ser, Adrien Bennetot, Siham Tabik,
Alberto Barbado, Salvador García,
Sergio Gil-López, Daniel Molina,
Richard Benjamins, et al . 
2020.

 
 Explainable Artificial Intelligence (XAI):
Concepts, taxonomies, opportunities and challenges toward responsible AI.

 
 Information fusion 58
(2020), 82–115.

 
 
 

 
 Banerjee and Lavie (2005) 
 
Satanjeev Banerjee and
Alon Lavie. 2005.

 
 METEOR: An Automatic Metric for MT Evaluation
with Improved Correlation with Human Judgments. In
 Proceedings of the ACL Workshop on Intrinsic and
Extrinsic Evaluation Measures for Machine Translation and/or Summarization .
Association for Computational Linguistics,
Ann Arbor, Michigan, 65–72.

 
 
 https://aclanthology.org/W05-0909 

 

 
 Barzilay and Lapata (2005) 
 
Regina Barzilay and
Mirella Lapata. 2005.

 
 Modeling Local Coherence: An Entity-Based
Approach. In Proceedings of the 43rd Annual
Meeting of the Association for Computational Linguistics (ACL’05) .
Association for Computational Linguistics,
Ann Arbor, Michigan, 141–148.

 
 
 https://doi.org/10.3115/1219840.1219858 

 

 
 Ben Abacha and Demner-Fushman (2019) 
 
Asma Ben Abacha and Dina
Demner-Fushman. 2019.

 
 On the Role of Question Summarization and
Information Source Restriction in Consumer Health Question Answering.

 
 AMIA Joint Summits on Translational Science
proceedings. AMIA Joint Summits on Translational Science 
2019 (05 2019),
117–126.

 
 
 

 
 Ben Abacha et al . (2021) 
 
Asma Ben Abacha, Yassine
Mrabet, Yuhao Zhang, Chaitanya Shivade,
Curtis Langlotz, and Dina
Demner-Fushman. 2021.

 
 Overview of the MEDIQA 2021 Shared Task on
Summarization in the Medical Domain. In
 Proceedings of the 20th Workshop on Biomedical
Language Processing . Association for Computational
Linguistics, Online, 74–85.

 
 
 https://doi.org/10.18653/v1/2021.bionlp-1.8 

 

 
 Bhandari et al . (2020) 
 
Manik Bhandari,
Pranav Narayan Gour, Atabak Ashfaq,
Pengfei Liu, and Graham Neubig.
2020.

 
 Re-evaluating Evaluation in Text Summarization. In
 Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP) .
Association for Computational Linguistics,
Online, 9347–9359.

 
 
 https://doi.org/10.18653/v1/2020.emnlp-main.751 

 

 
 Bian and Xie (2021) 
 
Yuemin Bian and
Xiang-Qun Xie. 2021.

 
 Generative chemistry: drug discovery with deep
learning generative models.

 
 Journal of Molecular Modeling 
27, 3 (2021),
1–18.

 
 
 

 
 Bigeard et al . (2015) 
 
Elise Bigeard, Vianney
Jouhet, Fleur Mougin, Frantz Thiessard,
and Natalia Grabar. 2015.

 
 Automatic extraction of numerical values from
unstructured data in EHRs.

 
 Studies in health technology and
informatics 210 (05
2015), 50–4.

 
 
 https://doi.org/10.3233/978-1-61499-512-8-50 

 

 
 Black and Colford (2017) 
 
Meghan Black and Cristin
Colford. 2017.

 
 Transitions of Care: Improving the Quality of
Discharge Summaries Completed By Internal Medicine Residents.

 
 MedEdPORTAL Publications 
13 (08 2017).

 
 
 https://doi.org/10.15766/mep_2374-8265.10613 

 

 
 Bodenreider (2004) 
 
Olivier Bodenreider.
2004.

 
 The Unified Medical Language System (UMLS):
Integrating Biomedical Terminology.

 
 Nucleic acids research 
32 (02 2004),
D267–70.

 
 
 https://doi.org/10.1093/nar/gkh061 

 

 
 Bontcheva et al . (2013) 
 
Kalina Bontcheva,
Genevieve Gorrell, and Bridgette
Wessels. 2013.

 
 Social media and information overload: Survey
results.

 
 arXiv preprint arXiv:1306.0813 
(2013).

 
 
 

 
 Bose et al . (2021) 
 
Priyankar Bose, Sriram
Srinivasan, William C. Sleeman, Jatinder
Palta, Rishabh Kapoor, and Preetam
Ghosh. 2021.

 
 A Survey on Recent Named Entity Recognition and
Relationship Extraction Techniques on Clinical Texts.

 
 Applied Sciences 11,
18 (2021).

 
 

 https://doi.org/10.3390/app11188319 

 

 
 Brown et al . (2020) 
 
Tom B. Brown, Benjamin
Mann, Nick Ryder, Melanie Subbiah,
Jared Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam,
Girish Sastry, Amanda Askell,
Sandhini Agarwal, Ariel Herbert-Voss,
Gretchen Krueger, Tom Henighan,
Rewon Child, Aditya Ramesh,
Daniel M. Ziegler, Jeffrey Wu,
Clemens Winter, Christopher Hesse,
Mark Chen, Eric Sigler,
Mateusz Litwin, Scott Gray,
Benjamin Chess, Jack Clark,
Christopher Berner, Sam McCandlish,
Alec Radford, Ilya Sutskever, and
Dario Amodei. 2020.

 
 Language Models are Few-Shot Learners.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2005.14165 

 

 
 Cai et al . (2022a) 
 
Xiaoyan Cai, Sen Liu,
Libin Yang, Yan Lu,
Jintao Zhao, Dinggang Shen, and
Tianming Liu. 2022a.

 
 COVIDSum: A linguistically enriched SciBERT-based
summarization model for COVID-19 scientific papers.

 
 Journal of Biomedical Informatics 
127 (2022), 103999.

 
 
 

 
 Cai et al . (2022b) 
 
Xiaoyan Cai, Sen Liu,
Libin Yang, Yan Lu,
Jintao Zhao, Dinggang Shen, and
Tianming Liu. 2022b.

 
 COVIDSum: A Linguistically Enriched SciBERT-Based
Summarization Model for COVID-19 Scientific Papers.

 
 J. of Biomedical Informatics 
127, C (mar
2022), 13 pages.

 
 

 https://doi.org/10.1016/j.jbi.2022.103999 

 

 
 Castelvecchi (2016) 
 
Davide Castelvecchi.
2016.

 
 Can we open the black box of AI?

 
 Nature News 538,
7623 (2016), 20.

 
 
 

 
 Chen et al . (2019) 
 
Irene Y Chen, Peter
Szolovits, and Marzyeh Ghassemi.
2019.

 
 Can AI help reduce disparities in general medical
and mental health care?

 
 AMA journal of ethics 21,
2 (2019), 167–179.

 
 
 

 
 Chen et al . (2022) 
 
Xiuying Chen, Mingzhe Li,
Xin Gao, and Xiangliang Zhang.
2022.

 
 Towards Improving Faithfulness in Abstractive
Summarization.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2210.01877 

 

 
 Chintagunta et al . (2021a) 
 
Bharath Chintagunta, Namit
Katariya, Xavier Amatriain, and Anitha
Kannan. 2021a.

 
 Medically Aware GPT-3 as a Data Generator for
Medical Dialogue Summarization. In Proceedings of
the Second Workshop on Natural Language Processing for Medical
Conversations . Association for Computational
Linguistics, Online, 66–76.

 
 
 https://doi.org/10.18653/v1/2021.nlpmc-1.9 

 

 
 Chintagunta et al . (2021b) 
 
Bharath Chintagunta, Namit
Katariya, Xavier Amatriain, and Anitha
Kannan. 2021b.

 
 Medically aware gpt-3 as a data generator for
medical dialogue summarization. In Machine
Learning for Healthcare Conference . PMLR, 354–372.

 
 
 

 
 Clark et al . (2019) 
 
Elizabeth Clark, Asli
Celikyilmaz, and Noah A. Smith.
2019.

 
 Sentence Mover’s Similarity: Automatic Evaluation
for Multi-Sentence Texts. In Proceedings of the
57th Annual Meeting of the Association for Computational Linguistics .
Association for Computational Linguistics,
Florence, Italy, 2748–2760.

 
 
 https://doi.org/10.18653/v1/P19-1264 

 

 
 Coleman and Liau (1975) 
 
Meri Coleman and Ta Lin
Liau. 1975.

 
 A computer readability formula designed for machine
scoring.

 
 Journal of Applied Psychology 
60, 2 (1975),
283.

 
 
 

 
 Correa et al . (2021) 
 
Ramon Correa,
Jiwoong Jason Jeong, Bhavik Patel,
Hari Trivedi, Judy W. Gichoya, and
Imon Banerjee. 2021.

 
 Two-step adversarial debiasing with partial learning
– medical image case-studies.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2111.08711 

 

 
 Delbrouck et al . (2021) 
 
Jean-Benoit Delbrouck,
Cassie Zhang, and Daniel Rubin.
2021.

 
 QIAI at MEDIQA 2021: Multimodal Radiology
Report Summarization. In Proceedings of the 20th
Workshop on Biomedical Language Processing . Association
for Computational Linguistics, Online,
285–290.

 
 
 https://doi.org/10.18653/v1/2021.bionlp-1.33 

 

 
 Demner-Fushman et al . (2022) 
 
Dina Demner-Fushman,
Kevin Bretonnel Cohen, Sophia Ananiadou,
and Junichi Tsujii (Eds.).
2022.

 
 Proceedings of the 21st Workshop on
Biomedical Language Processing . Association for
Computational Linguistics, Dublin, Ireland.

 
 
 https://aclanthology.org/2022.bionlp-1.0 

 

 
 Demner-Fushman et al . (2015) 
 
Dina Demner-Fushman, Marc
Kohli, Marc Rosenman, Sonya Shooshan,
Laritza Rodriguez, Sameer Antani,
George Thoma, and Clement Mcdonald.
2015.

 
 Preparing a collection of radiology examinations
for distribution and retrieval.

 
 Journal of the American Medical Informatics
Association : JAMIA 23 (07
2015).

 
 
 https://doi.org/10.1093/jamia/ocv080 

 

 
 DeYoung et al . (2021) 
 
Jay DeYoung, Iz Beltagy,
Madeleine van Zuylen, Bailey Kuehl, and
Lucy Lu Wang. 2021.

 
 MSˆ2: Multi-Document Summarization of
Medical Studies. In Proceedings of the 2021
Conference on Empirical Methods in Natural Language Processing .
Association for Computational Linguistics,
Online and Punta Cana, Dominican Republic,
7494–7513.

 
 
 https://aclanthology.org/2021.emnlp-main.594 

 

 
 Du et al . (2020) 
 
Yongping Du, Qingxiao Li,
Lulin Wang, and Yanqing He.
2020.

 
 Biomedical-domain pre-trained language model for
extractive summarization.

 
 Knowledge-Based Systems 
199 (2020), 105964.

 
 
 

 
 Enarvi et al . (2020) 
 
Seppo Enarvi, Marilisa
Amoia, Miguel Del-Agua Teba, Brian
Delaney, Frank Diehl, Stefan Hahn,
Kristina Harris, Liam McGrath,
Yue Pan, Joel Pinto, et al . 
2020.

 
 Generating medical reports from patient-doctor
conversations using sequence-to-sequence models. In
 Proceedings of the first workshop on natural
language processing for medical conversations . 22–30.

 
 
 

 
 Fabbri et al . (2020) 
 
Alexander R. Fabbri,
Wojciech Kryściński, Bryan McCann,
Caiming Xiong, Richard Socher, and
Dragomir Radev. 2020.

 
 SummEval: Re-evaluating Summarization Evaluation.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2007.12626 

 

 
 Feng et al . (2021) 
 
Xiachong Feng, Xiaocheng
Feng, and Bing Qin. 2021.

 
 A survey on dialogue summarization: Recent advances
and new frontiers.

 
 arXiv preprint arXiv:2107.03175 
(2021).

 
 
 

 
 Flory et al . (1992) 
 
Steven M Flory, Thomas J
Phillips Jr, and Maurice F Tassin.
1992.

 
 Measuring readability: A comparison of accounting
textbooks.

 
 Journal of Accounting Education 
10, 1 (1992),
151–161.

 
 
 

 
 Fort and Couillault (2016) 
 
Karën Fort and Alain
Couillault. 2016.

 
 Yes, we care! results of the ethics and natural
language processing surveys. In international
Language Resources and Evaluation Conference (LREC) 2016 .

 
 
 

 
 Garrido-Muñoz et al . (2021) 
 
Ismael Garrido-Muñoz,
Arturo Montejo-Ráez, Fernando
Martínez-Santiago, and L Alfonso
Ureña-López. 2021.

 
 A survey on bias in deep NLP.

 
 Applied Sciences 11,
7 (2021), 3184.

 
 
 

 
 Gayathri and Jaisankar (2015) 
 
P. Gayathri and N.
Jaisankar. 2015.

 
 Towards an Efficient Approach for Automatic Medical
Document Summarization.

 
 Cybernetics and Information Technologies 
15 (11 2015).

 
 
 https://doi.org/10.1515/cait-2015-0056 

 

 
 Gershanik et al . (2011) 
 
Esteban Gershanik, Ronilda
Lacson, and Ramin Khorasani.
2011.

 
 Critical Finding Capture in the Impression Section
of Radiology Reports.

 
 AMIA … Annual Symposium proceedings / AMIA
Symposium. AMIA Symposium 2011 (01
2011), 465–9.

 
 
 

 
 Goel et al . (2021) 
 
Karan Goel, Nazneen
Rajani, Jesse Vig, Samson Tan,
Jason Wu, Stephan Zheng,
Caiming Xiong, Mohit Bansal, and
Christopher Ré. 2021.

 
 Robustness Gym: Unifying the NLP Evaluation
Landscape.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2101.04840 

 

 
 Guo et al . (2020) 
 
Yue Guo, Wei Qiu,
Yizhong Wang, and Trevor Cohen.
2020.

 
 Automated Lay Language Summarization of Biomedical
Scientific Reviews.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2012.12573 

 

 
 Hariton and Locascio (2018) 
 
Eduardo Hariton and
Joseph J Locascio. 2018.

 
 Randomised controlled trials—the gold standard
for effectiveness research.

 
 BJOG: an international journal of obstetrics
and gynaecology 125, 13
(2018), 1716.

 
 
 

 
 Harkema et al . (2009) 
 
Henk Harkema, John N
Dowling, Tyler Thornblade, and Wendy W
Chapman. 2009.

 
 ConText: an algorithm for determining negation,
experiencer, and temporal status from clinical reports.

 
 Journal of biomedical informatics 
42, 5 (2009),
839–851.

 
 
 

 
 Hartung et al . (2020) 
 
Michael Hartung, Ian
Bickle, Frank Gaillard, and Jeffrey
Kanne. 2020.

 
 How to Create a Great Radiology Report.

 
 Radiographics 40
(10 2020), 1658–1670.

 
 
 https://doi.org/10.1148/rg.2020200020 

 

 
 Hu et al . (2021) 
 
Jinpeng Hu, Jianling Li,
Zhihong Chen, Yaling Shen,
Yan Song, Xiang Wan, and
Tsung-Hui Chang. 2021.

 
 Word graph guided summarization for radiology
findings.

 
 arXiv preprint arXiv:2112.09925 
(2021).

 
 
 

 
 Huang et al . (2011) 
 
Ling Huang, Anthony D
Joseph, Blaine Nelson, Benjamin IP
Rubinstein, and J Doug Tygar.
2011.

 
 Adversarial machine learning. In
 Proceedings of the 4th ACM workshop on Security and
artificial intelligence . 43–58.

 
 
 

 
 Hussain et al . (2021) 
 
Tanveer Hussain, Khan
Muhammad, Weiping Ding, Jaime Lloret,
Sung Wook Baik, and Victor Hugo C de
Albuquerque. 2021.

 
 A comprehensive survey of multi-view video
summarization.

 
 Pattern Recognition 109
(2021), 107567.

 
 
 

 
 Jain et al . (2022) 
 
Raghav Jain, Vaibhav
Mavi, Anubhav Jangra, and Sriparna
Saha. 2022.

 
 WIDAR - Weighted Input Document Augmented ROUGE.
In Advances in Information Retrieval: 44th European
Conference on IR Research, ECIR 2022, Stavanger, Norway, April 10–14, 2022,
Proceedings, Part I (Stavanger, Norway).
Springer-Verlag, Berlin, Heidelberg,
304–321.

 
 

 https://doi.org/10.1007/978-3-030-99736-6_21 

 

 
 Jangra et al . (2020a) 
 
Anubhav Jangra, Adam
Jatowt, Mohammad Hasanuzzaman, and
Sriparna Saha. (ECIR
2020) a.

 
 Text-image-video summary generation using joint
integer linear programming. In European Conference
on Information Retrieval .

 
 
 

 
 Jangra et al . (2021a) 
 
Anubhav Jangra, Adam
Jatowt, Sriparna Saha, and Mohammad
Hasanuzzaman. 2021a.

 
 A Survey on Multi-modal Summarization.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2109.05199 

 

 
 Jangra et al . (2020b) 
 
Anubhav Jangra, Sriparna
Saha, Adam Jatowt, and Mohammad
Hasanuzzaman. (SIGIR 2020) b.

 
 Multi-Modal Summary Generation Using
Multi-Objective Optimization. In Proceedings of
the 43rd International ACM SIGIR Conference on Research and Development in
Information Retrieval .

 
 
 

 
 Jangra et al . (2021b) 
 
Anubhav Jangra, Sriparna
Saha, Adam Jatowt, and Mohammed
Hasanuzzaman. (SIGIR 2021) b.

 
 Multi-Modal Supplementary-Complementary
Summarization Using Multi-Objective Optimization. In
 Proceedings of the 44th International ACM SIGIR
Conference on Research and Development in Information Retrieval .

 
 
 

 
 Jin et al . (2022) 
 
Qiao Jin, Zheng Yuan,
Guangzhi Xiong, Qianlan Yu,
Huaiyuan Ying, Chuanqi Tan,
Mosha Chen, Songfang Huang,
Xiaozhong Liu, and Sheng Yu.
2022.

 
 Biomedical Question Answering: A Survey of
Approaches and Challenges.

 
 ACM Comput. Surv. 55,
2, Article 35 (jan
2022), 36 pages.

 
 

 https://doi.org/10.1145/3490238 

 

 
 Jo et al . (2019) 
 
Heuisug Jo, Keeho Park,
and Su Mi Jung. 2019.

 
 A scoping review of consumer needs for cancer
information.

 
 Patient education and counseling 
102 7 (2019), 1237–1250.

 
 
 

 
 Johnson et al . (2016) 
 
Alistair EW Johnson, Tom J
Pollard, Lu Shen, Li-wei H Lehman,
Mengling Feng, Mohammad Ghassemi,
Benjamin Moody, Peter Szolovits,
Leo Anthony Celi, and Roger G Mark.
2016.

 
 MIMIC-III, a freely accessible critical care
database.

 
 Scientific data 3,
1 (2016), 1–9.

 
 
 

 
 Johnson et al . (2019) 
 
Alistair E. W. Johnson,
Tom J. Pollard, Nathaniel R. Greenbaum,
Matthew P. Lungren, Chih-ying Deng,
Yifan Peng, Zhiyong Lu,
Roger G. Mark, Seth J. Berkowitz, and
Steven Horng. 2019.

 
 MIMIC-CXR-JPG, a large publicly available database of
labeled chest radiographs.

 
 
 
 
 https://doi.org/10.48550/ARXIV.1901.07042 

 

 
 Joshi et al . (2020) 
 
Anirudh Joshi, Namit
Katariya, Xavier Amatriain, and Anitha
Kannan. 2020.

 
 Dr. Summarize: Global Summarization of Medical
Dialogue by Exploiting Local Structures.. In
 Findings of the Association for Computational
Linguistics: EMNLP 2020 . Association for Computational
Linguistics, Online, 3755–3763.

 
 
 https://doi.org/10.18653/v1/2020.findings-emnlp.335 

 

 
 Kalyan and Sangeetha (2020) 
 
Katikapalli Subramanyam Kalyan and
Sivanesan Sangeetha. 2020.

 
 SECNLP: A survey of embeddings in clinical natural
language processing.

 
 Journal of biomedical informatics 
101 (2020), 103323.

 
 
 

 
 Kaye (2012) 
 
Jane Kaye.
2012.

 
 The tension between data sharing and the protection
of privacy in genomics research.

 
 Annual review of genomics and human
genetics 13 (2012),
415.

 
 
 

 
 Kedzie et al . (2018) 
 
Chris Kedzie, Kathleen
McKeown, and Hal Daume III.
2018.

 
 Content selection in deep learning models of
summarization.

 
 arXiv preprint arXiv:1810.12343 
(2018).

 
 
 

 
 Kieuvongngam et al . (2020) 
 
Virapat Kieuvongngam,
Bowen Tan, and Yiming Niu.
2020.

 
 Automatic Text Summarization of COVID-19 Medical
Research Articles using BERT and GPT-2.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2006.01997 

 

 
 Kincaid et al . (1975) 
 
J Peter Kincaid, Robert P
Fishburne Jr, Richard L Rogers, and
Brad S Chissom. 1975.

 
 Derivation of new readability formulas
(automated readability index, fog count and flesch reading ease formula) for
navy enlisted personnel .

 
 Technical Report. Naval
Technical Training Command Millington TN Research Branch.

 
 
 

 
 Kondadadi et al . (2021) 
 
Ravikumar Kondadadi, Sahil
Manchanda, Jason Ngo, and Ronan
McCormack. 2021.

 
 Optum at MEDIQA 2021: Abstractive Summarization of
Radiology Reports using simple BART Finetuning. In
 Proceedings of the 20th Workshop on Biomedical
Language Processing . 280–284.

 
 
 

 
 Kripalani et al . (2007) 
 
Sunil Kripalani, Frank
Lefevre, Christopher Phillips, Mark
Williams, Preetha Basaviah, and David
Baker. 2007.

 
 Deficits in Communication and Information Transfer
Between Hospital-Based and Primary Care Physicians.

 
 JAMA : the journal of the American Medical
Association 297 (03
2007), 831–41.

 
 
 https://doi.org/10.1001/jama.297.8.831 

 

 
 Krishna et al . (2021) 
 
Kundan Krishna, Sopan
Khosla, Jeffrey Bigham, and Zachary C.
Lipton. 2021.

 
 Generating SOAP Notes from Doctor-Patient
Conversations Using Modular Summarization Techniques. In
 Proceedings of the 59th Annual Meeting of the
Association for Computational Linguistics and the 11th International Joint
Conference on Natural Language Processing (Volume 1: Long Papers) .
Association for Computational Linguistics,
Online, 4958–4972.

 
 
 https://doi.org/10.18653/v1/2021.acl-long.384 

 

 
 Kryscinski et al . (2019) 
 
Wojciech Kryscinski,
Nitish Shirish Keskar, Bryan McCann,
Caiming Xiong, and Richard Socher.
2019.

 
 Neural Text Summarization: A Critical Evaluation.
In Proceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th International Joint
Conference on Natural Language Processing (EMNLP-IJCNLP) .
Association for Computational Linguistics,
Hong Kong, China, 540–551.

 
 
 https://doi.org/10.18653/v1/D19-1051 

 

 
 Lewis et al . (2019) 
 
Mike Lewis, Yinhan Liu,
Naman Goyal, Marjan Ghazvininejad,
Abdelrahman Mohamed, Omer Levy,
Ves Stoyanov, and Luke Zettlemoyer.
2019.

 
 BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension.

 
 
 
 
 https://doi.org/10.48550/ARXIV.1910.13461 

 

 
 Lin (2004) 
 
Chin-Yew Lin.
2004.

 
 ROUGE: A Package for Automatic Evaluation of
Summaries. In Text Summarization Branches Out .
Association for Computational Linguistics,
Barcelona, Spain, 74–81.

 
 
 https://aclanthology.org/W04-1013 

 

 
 Litvak and Vanetik (2017) 
 
Marina Litvak and
Natalia Vanetik. 2017.

 
 Query-based summarization using MDL principle.

 
 
 https://doi.org/10.18653/v1/W17-1004 

 

 
 Liu et al . (2021) 
 
Xiangbin Liu, Liping
Song, Shuai Liu, and Yudong Zhang.
2021.

 
 A review of deep-learning-based medical image
segmentation methods.

 
 Sustainability 13,
3 (2021), 1224.

 
 
 

 
 Liu et al . (2019) 
 
Zhengyuan Liu, Angela Ng,
Sheldon Lee Shao Guang, AiTi Aw, and
Nancy F. Chen. 2019.

 
 Topic-Aware Pointer-Generator Networks for
Summarizing Spoken Conversations.

 
 2019 IEEE Automatic Speech Recognition and
Understanding Workshop (ASRU) (2019),
814–821.

 
 
 

 
 Luo et al . (2022) 
 
Luyang Luo, Dunyuan Xu,
Hao Chen, Tien-Tsin Wong, and
Pheng-Ann Heng. 2022.

 
 Pseudo Bias-Balanced Learning for Debiased Chest
X-ray Classification.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2203.09860 

 

 
 Ma et al . (2020) 
 
Congbo Ma, Wei Emma
Zhang, Mingyu Guo, Hu Wang, and
Quan Z Sheng. 2020.

 
 Multi-document summarization via deep learning
techniques: A survey.

 
 ACM Computing Surveys (CSUR) 
(2020).

 
 
 

 
 MacAvaney et al . (2019) 
 
Sean MacAvaney, Sajad
Sotudeh, Arman Cohan, Nazli Goharian,
Ish Talati, and Ross W Filice.
2019.

 
 Ontology-aware clinical abstractive summarization.
In Proceedings of the 42nd International ACM SIGIR
Conference on Research and Development in Information Retrieval .
1013–1016.

 
 
 

 
 Mann et al . (2020) 
 
Devin M. Mann, Ji Chen,
Rumi Chunara, Paul A. Testa, and
Oded Nov. 2020.

 
 COVID-19 transforms health care through
telemedicine: Evidence from the field.

 
 Journal of the American Medical Informatics
Association : JAMIA 27 (2020),
1132 – 1135.

 
 
 

 
 Maynez et al . (2020a) 
 
Joshua Maynez, Shashi
Narayan, Bernd Bohnet, and Ryan
McDonald. 2020a.

 
 On faithfulness and factuality in abstractive
summarization.

 
 arXiv preprint arXiv:2005.00661 
(2020).

 
 
 

 
 Maynez et al . (2020b) 
 
Joshua Maynez, Shashi
Narayan, Bernd Bohnet, and Ryan
McDonald. 2020b.

 
 On Faithfulness and Factuality in Abstractive
Summarization.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2005.00661 

 

 
 Moramarco et al . (2021) 
 
Francesco Moramarco, Damir
Juric, Aleksandar Savkov, and Ehud
Reiter. 2021.

 
 Towards objectively evaluating the quality of
generated medical summaries.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2104.04412 

 

 
 Moravvej et al . (2021) 
 
Seyed Vahid Moravvej,
Abdolreza Mirzaei, and Mehran
Safayani. 2021.

 
 Biomedical text summarization using Conditional
Generative Adversarial Network(CGAN).

 
 
 
 
 https://doi.org/10.48550/ARXIV.2110.11870 

 

 
 Mrini et al . (2021a) 
 
Khalil Mrini, Franck
Dernoncourt, Walter Chang, Emilia
Farcas, and Ndapa Nakashole.
2021a.

 
 Joint Summarization-Entailment Optimization for
Consumer Health Question Understanding.

 
 Proceedings of the Second Workshop on Natural
Language Processing for Medical Conversations (2021).

 
 
 

 
 Mrini et al . (2021b) 
 
Khalil Mrini, Franck
Dernoncourt, Seunghyun Yoon, Trung Bui,
Walter Chang, Emilia Farcas, and
Ndapandula Nakashole. 2021b.

 
 A gradually soft multi-task and data-augmented
approach to medical question understanding. In
 Proceedings of the 59th Annual Meeting of the
Association for Computational Linguistics and the 11th International Joint
Conference on Natural Language Processing (Volume 1: Long Papers) .
1505–1515.

 
 
 

 
 Mrini et al . (2021c) 
 
Khalil Mrini, Franck
Dernoncourt, Seunghyun Yoon, Trung H.
Bui, Walter Chang, Emilia Farcas, and
Ndapandula Nakashole. 2021c.

 
 A Gradually Soft Multi-Task and Data-Augmented
Approach to Medical Question Understanding. In
 ACL .

 
 
 

 
 Nentidis et al . (2021) 
 
Anastasios Nentidis,
Georgios Katsimpras, Eirini Vandorou,
Anastasia Krithara, Luis Gasco,
Martin Krallinger, and Georgios
Paliouras. 2021.

 
 Overview of BioASQ 2021: The Ninth BioASQ
Challenge on Large-Scale Biomedical Semantic Indexing and Question
Answering.

 
 In Lecture Notes in Computer Science .
Springer International Publishing,
239–263.

 
 
 https://doi.org/10.1007/978-3-030-85251-1_18 

 

 
 November (2012) 
 
Joseph A November.
2012.

 
 Biomedical computing: Digitizing life in
the United States . Vol. 130.

 
 JHU Press.

 
 
 

 
 Olatunji et al . (2022) 
 
Iyiola E Olatunji, Jens
Rauch, Matthias Katzensteiner, and
Megha Khosla. 2022.

 
 A review of anonymization for healthcare data.

 
 Big Data (2022).

 
 
 

 
 Pandey et al . (2021) 
 
Babita Pandey,
Devendra Kumar Pandey, Brijendra Pratap
Mishra, and Wasiur Rhmann.
2021.

 
 A comprehensive survey of deep learning in the
field of medical imaging and medical natural language processing: Challenges
and research directions.

 
 Journal of King Saud University-Computer and
Information Sciences (2021).

 
 
 

 
 Papineni et al . (2002) 
 
Kishore Papineni, Salim
Roukos, Todd Ward, and Wei-Jing Zhu.
2002.

 
 Bleu: a method for automatic evaluation of machine
translation. In Proceedings of the 40th annual
meeting of the Association for Computational Linguistics .
311–318.

 
 
 

 
 Park (2020) 
 
Jong Won Park.
2020.

 
 Continual BERT: Continual Learning for Adaptive
Extractive Summarization of COVID-19 Literature.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2007.03405 

 

 
 Pasquali et al . (2021) 
 
Arian Pasquali, Ricardo
Campos, Alexandre Ribeiro,
Brenda Salenave Santana, Alípio
Jorge, and Adam Jatowt.
2021.

 
 TLS-Covid19: A New Annotated Corpus for Timeline
Summarization. In Advances in Information
Retrieval - 43rd European Conference on IR Research, ECIR 2021, Virtual
Event, March 28 - April 1, 2021, Proceedings, Part I 
 (Lecture Notes in Computer Science,
Vol. 12656) . Springer,
497–512.

 
 
 

 
 Portoghese et al . (2014) 
 
Igor Portoghese, Maura
Galletta, Rosa Cristina Coppola, Gabriele
Finco, and Marcello Campagna.
2014.

 
 Burnout and workload among health care workers: the
moderating role of job control.

 
 Safety and health at work 
5, 3 (2014),
152–157.

 
 
 

 
 Priya et al . (2017) 
 
R Priya, S Sivasankaran,
P Ravisasthiri, and S Sivachandiran.
2017.

 
 A survey on security attacks in electronic
healthcare systems. In 2017 international
conference on communication and signal processing (ICCSP) . IEEE,
0691–0694.

 
 
 

 
 Raffel et al . (2019) 
 
Colin Raffel, Noam
Shazeer, Adam Roberts, Katherine Lee,
Sharan Narang, Michael Matena,
Yanqi Zhou, Wei Li, and
Peter J. Liu. 2019.

 
 Exploring the Limits of Transfer Learning with a
Unified Text-to-Text Transformer.

 
 
 
 
 https://doi.org/10.48550/ARXIV.1910.10683 

 

 
 Rahman and Borah (2019) 
 
Nazreena Rahman and
Bhogeswar Borah. 2019.

 
 Improvement of query-based text summarization using
word sense disambiguation.

 
 Complex Intelligent Systems 
6 (07 2019).

 
 
 https://doi.org/10.1007/s40747-019-0115-2 

 

 
 Rao and Daumé (2019) 
 
Sudha Rao and Hal
Daumé. 2019.

 
 Answer-based Adversarial Training for Generating
Clarification Questions.

 
 
 
 
 https://doi.org/10.48550/ARXIV.1904.02281 

 

 
 Ribeiro et al . (2016) 
 
Marco Tulio Ribeiro,
Sameer Singh, and Carlos Guestrin.
2016.

 
 "Why Should I Trust You?": Explaining the Predictions
of Any Classifier.

 
 
 
 
 https://doi.org/10.48550/ARXIV.1602.04938 

 

 
 Roberts and Demner-Fushman (2016) 
 
Kirk Roberts and Dina
Demner-Fushman. 2016.

 
 Interactive use of online health resources: a
comparison of consumer and professional questions.

 
 Journal of the American Medical Informatics
Association 23, 4 (05
2016), 802–811.

 
 

 https://doi.org/10.1093/jamia/ocw024 

 

 
 Rutten et al . (2019) 
 
Lila J. Finney Rutten,
Kelly D. Blake, Alexandra J.
Greenberg-Worisek, Summer V. Allen,
Richard P. Moser, and Bradford William
Hesse. 2019.

 
 Online Health Information Seeking Among US Adults:
Measuring Progress Toward a Healthy People 2020 Objective.

 
 Public Health Reports 
134 (2019), 617 – 625.

 
 
 

 
 Sarkar (2009) 
 
K. Sarkar.
2009.

 
 Using Domain Knowledge for Text Summarization in
Medical Domain.

 
 
 

 
 Sarkar et al . (2011) 
 
Kamal Sarkar, Mita
Nasipuri, and Suranjan Ghose.
2011.

 
 Using machine learning for medical document
summarization.

 
 International Journal of Database Theory and
Application 4, 1
(2011), 31–48.

 
 
 

 
 Sarker (2021) 
 
Iqbal H Sarker.
2021.

 
 Machine learning: Algorithms, real-world
applications and research directions.

 
 SN Computer Science 2,
3 (2021), 1–21.

 
 
 

 
 Savery et al . (2020) 
 
Max Savery, Asma
Ben Abacha, Soumya Gayen, and Dina
Demner-Fushman. 2020.

 
 Question-Driven Summarization of Answers to Consumer
Health Questions.

 
 
 
 
 

 
 Shah et al . (2021a) 
 
Darsh Shah, Lili Yu,
Tao Lei, and Regina Barzilay.
2021a.

 
 Nutri-bullets Hybrid: Consensual Multi-document
Summarization. In Proceedings of the 2021
Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies . Association
for Computational Linguistics, Online,
5213–5222.

 
 
 https://doi.org/10.18653/v1/2021.naacl-main.411 

 

 
 Shah et al . (2021b) 
 
Darsh Shah, Lili Yu,
Tao Lei, and Regina Barzilay.
2021b.

 
 Nutri-bullets Hybrid: Consensual Multi-document
Summarization. In Proceedings of the 2021
Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies . 5213–5222.

 
 
 

 
 Shah et al . (2021c) 
 
Darsh J Shah, Lili Yu,
Tao Lei, and Regina Barzilay.
2021c.

 
 Nutri-bullets: Summarizing Health Studies by
Composing Segments.

 
 (2021).

 
 
 https://doi.org/10.48550/ARXIV.2103.11921 

 

 
 Sharma et al . (2022) 
 
Deepak Kumar Sharma,
Mayukh Chatterjee, Gurmehak Kaur, and
Suchitra Vavilala. 2022.

 
 3 - Deep learning applications for disease
diagnosis.

 
 In Deep Learning for Medical Applications
with Unique Data , Deepak Gupta,
Utku Kose, Ashish Khanna, and
Valentina Emilia Balas (Eds.).
Academic Press, 31–51.

 
 

 https://doi.org/10.1016/B978-0-12-824145-5.00005-8 

 

 
 Shing et al . (2021) 
 
Han-Chin Shing, Chaitanya
Shivade, Nima Pourdamghani, Feng Nan,
Philip Resnik, Douglas Oard, and
Parminder Bhatia. 2021.

 
 Towards Clinical Encounter Summarization: Learning to
Compose Discharge Summaries from Prior Notes.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2104.13498 

 

 
 Soldaini (2016) 
 
Luca Soldaini.
2016.

 
 QuickUMLS: a Fast, Unsupervised Approach for
Medical Concept Extraction.

 
 
 

 
 Song et al . (2020a) 
 
Yan Song, Yuanhe Tian,
Nan Wang, and Fei Xia.
2020a.

 
 Summarizing Medical Conversations via Identifying
Important Utterances. In Proceedings of the 28th
International Conference on Computational Linguistics .
International Committee on Computational Linguistics,
Barcelona, Spain (Online), 717–729.

 
 
 https://doi.org/10.18653/v1/2020.coling-main.63 

 

 
 Song et al . (2020b) 
 
Yan Song, Yuanhe Tian,
Nan Wang, and Fei Xia.
2020b.

 
 Summarizing medical conversations via identifying
important utterances. In Proceedings of the 28th
International Conference on Computational Linguistics .
717–729.

 
 
 

 
 Sotudeh et al . (2020) 
 
Sajad Sotudeh, Nazli
Goharian, and Ross W Filice.
2020.

 
 Attend to medical ontologies: Content selection for
clinical abstractive summarization.

 
 arXiv preprint arXiv:2005.00163 
(2020).

 
 
 

 
 Sundararajan et al . (2017) 
 
Mukund Sundararajan, Ankur
Taly, and Qiqi Yan. 2017.

 
 Axiomatic Attribution for Deep Networks.

 
 
 
 
 https://doi.org/10.48550/ARXIV.1703.01365 

 

 
 Thawani et al . (2021a) 
 
Avijit Thawani, Jay
Pujara, and Filip Ilievski.
2021a.

 
 Numeracy enhances the Literacy of Language Models.
In Proceedings of the 2021 Conference on Empirical
Methods in Natural Language Processing . Association for
Computational Linguistics, Online and Punta Cana,
Dominican Republic, 6960–6967.

 
 
 https://doi.org/10.18653/v1/2021.emnlp-main.557 

 

 
 Thawani et al . (2021b) 
 
Avijit Thawani, Jay
Pujara, Filip Ilievski, and Pedro
Szekely. 2021b.

 
 Representing Numbers in NLP: a Survey and a
Vision. In Proceedings of the 2021 Conference of
the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies . Association for
Computational Linguistics, Online,
644–656.

 
 
 https://doi.org/10.18653/v1/2021.naacl-main.53 

 

 
 Tsatsaronis et al . (2015) 
 
George Tsatsaronis,
Georgios Balikas, Prodromos Malakasiotis,
Ioannis Partalas, Matthias Zschunke,
Michael Alvers, Dirk Weißenborn,
Anastasia Krithara, Sergios Petridis,
Dimitris Polychronopoulos, Yannis
Almirantis, John Pavlopoulos, Nicolas
Baskiotis, Patrick Gallinari, Thierry
Artieres, Axel-Cyrille Ngonga Ngomo,
Norman Heino, Eric Gaussier,
Liliana Barrio-Alvers, and Georgios
Paliouras. 2015.

 
 An overview of the BIOASQ large-scale biomedical
semantic indexing and question answering competition.

 
 BMC Bioinformatics 16
(04 2015), 138.

 
 
 https://doi.org/10.1186/s12859-015-0564-6 

 

 
 Valizadeh and Parde (2022) 
 
Mina Valizadeh and
Natalie Parde. 2022.

 
 The AI Doctor Is In: A Survey of Task-Oriented
Dialogue Systems for Healthcare Applications. In
 Proceedings of the 60th Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers) .
Association for Computational Linguistics,
Dublin, Ireland, 6638–6660.

 
 
 https://doi.org/10.18653/v1/2022.acl-long.458 

 

 
 Wallace et al . (2020) 
 
Byron C. Wallace, Sayantan
Saha, Frank Soboczenski, and Iain J.
Marshall. 2020.

 
 Generating (Factual?) Narrative Summaries of RCTs:
Experiments with Neural Multi-Document Summarization.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2008.11293 

 

 
 Wang et al . (2022) 
 
Jiaan Wang, Fandong Meng,
Duo Zheng, Yunlong Liang,
Zhixu Li, Jianfeng Qu, and
Jie Zhou. 2022.

 
 A survey on cross-lingual summarization.

 
 arXiv preprint arXiv:2203.12515 
(2022).

 
 
 

 
 Wang et al . (2020a) 
 
Lucy Lu Wang, Kyle Lo,
Yoganand Chandrasekhar, Russell Reas,
Jiangjiang Yang, Darrin Eide,
Kathryn Funk, Rodney Kinney,
Ziyang Liu, William Merrill,
et al . 2020a.

 
 Cord-19: The covid-19 open research dataset.

 
 ArXiv (2020).

 
 
 

 
 Wang et al . (2020b) 
 
Wenjie Wang, Youngja
Park, Taesung Lee, Ian Molloy,
Pengfei Tang, and Li Xiong.
2020b.

 
 Utilizing Multimodal Feature Consistency to Detect
Adversarial Examples on Clinical Summaries. In
 Proceedings of the 3rd Clinical Natural Language
Processing Workshop . Association for Computational
Linguistics, Online, 259–268.

 
 
 https://doi.org/10.18653/v1/2020.clinicalnlp-1.29 

 

 
 Wu et al . (2020) 
 
Stephen Wu, Kirk Roberts,
Surabhi Datta, Jingcheng Du,
Zongcheng Ji, Yuqi Si,
Sarvesh Soni, Qiong Wang,
Qiang Wei, Yang Xiang, et al . 
2020.

 
 Deep learning in clinical natural language
processing: a methodical review.

 
 Journal of the American Medical Informatics
Association 27, 3
(2020), 457–470.

 
 
 

 
 Xie et al . (2019) 
 
Yuan Xie, Bin Jiang,
Enhao Gong, Ying Li,
Guangming Zhu, Patrik Michel,
Max Wintermark, and Greg Zaharchuk.
2019.

 
 Use of gradient boosting machine learning to
predict patient outcome in acute ischemic stroke on the basis of imaging,
demographic, and clinical information.

 
 American Journal of Roentgenology 
212, 1 (2019),
44–51.

 
 
 

 
 Xu and Durrett (2021) 
 
Jiacheng Xu and Greg
Durrett. 2021.

 
 Dissecting Generation Modes for Abstractive
Summarization Models via Ablation and Attribution.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2106.01518 

 

 
 Yadav et al . (2022) 
 
Shweta Yadav, Deepak
Gupta, Asma Ben Abacha, and Dina
Demner-Fushman. 2022.

 
 Question-aware transformer models for consumer
health question summarization.

 
 Journal of Biomedical Informatics 
128 (2022), 104040.

 
 
 

 
 Yang et al . (2022) 
 
Jenny Yang, Andrew
Soltan, Yang Yang, and David Clifton.
2022.

 
 Algorithmic Fairness and Bias Mitigation for Clinical
Machine Learning: Insights from Rapid COVID-19 Diagnosis by Adversarial
Learning.

 
 
 
 
 https://doi.org/10.1101/2022.01.13.22268948 

 

 
 Zeng et al . (2020) 
 
Guangtao Zeng, Wenmian
Yang, Zeqian Ju, Yue Yang,
Sicheng Wang, Ruisi Zhang,
Meng Zhou, Jiaqi Zeng,
Xiangyu Dong, Ruoyu Zhang,
Hongchao Fang, Penghui Zhu,
Shu Chen, and Pengtao Xie.
2020.

 
 MedDialog: Large-scale Medical Dialogue
Datasets. In Proceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing (EMNLP) .
Association for Computational Linguistics,
Online, 9241–9250.

 
 
 https://doi.org/10.18653/v1/2020.emnlp-main.743 

 

 
 Zhang et al . (2022) 
 
Chunyan Zhang, Junchao
Wang, Qinglei Zhou, Ting Xu,
Ke Tang, Hairen Gui, and
Fudong Liu. 2022.

 
 A Survey of Automatic Source Code Summarization.

 
 Symmetry 14,
3 (2022), 471.

 
 
 

 
 Zhang et al . (2020a) 
 
Haoran Zhang, Amy X. Lu,
Mohamed Abdalla, Matthew McDermott, and
Marzyeh Ghassemi. 2020a.

 
 Hurtful Words: Quantifying Biases in Clinical
Contextual Word Embeddings.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2003.11515 

 

 
 Zhang et al . (2020b) 
 
Jingqing Zhang, Yao Zhao,
Mohammad Saleh, and Peter Liu.
2020b.

 
 Pegasus: Pre-training with extracted gap-sentences
for abstractive summarization. In International
Conference on Machine Learning . PMLR, 11328–11339.

 
 
 

 
 Zhang et al . (2021) 
 
Longxiang Zhang, Renato
Negrinho, Arindam Ghosh, Vasudevan
Jagannathan, Hamid Reza Hassanzadeh,
Thomas Schaaf, and Matthew R. Gormley.
2021.

 
 Leveraging Pretrained Models for Automatic
Summarization of Doctor-Patient Conversations.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2109.12174 

 

 
 Zhang et al . (2019a) 
 
Tianyi Zhang, Varsha
Kishore, Felix Wu, Kilian Q Weinberger,
and Yoav Artzi. 2019a.

 
 Bertscore: Evaluating text generation with bert.

 
 arXiv preprint arXiv:1904.09675 
(2019).

 
 
 

 
 Zhang et al . (2018) 
 
Yuhao Zhang, Daisy Yi
Ding, Tianpei Qian, Christopher D
Manning, and Curtis P Langlotz.
2018.

 
 Learning to summarize radiology findings.

 
 arXiv preprint arXiv:1809.04698 
(2018).

 
 
 

 
 Zhang et al . (2019b) 
 
Yuhao Zhang, Derek Merck,
Emily Bao Tsai, Christopher D Manning,
and Curtis P Langlotz. 2019b.

 
 Optimizing the factual correctness of a summary: A
study of summarizing radiology reports.

 
 arXiv preprint arXiv:1911.02541 
(2019).

 
 
 

 
 Zhou et al . (2021) 
 
Binggui Zhou, Guanghua
Yang, Zheng Shi, and Shaodan Ma.
2021.

 
 Natural Language Processing for Smart Healthcare.

 
 
 
 
 https://doi.org/10.48550/ARXIV.2110.15803 

 

 
 Zouhar et al . (2022) 
 
Vilém Zouhar, Marius
Mosbach, Debanjali Biswas, and Dietrich
Klakow. 2022.

 
 Artefact retrieval: Overview of NLP models with
knowledge base access.

 
 arXiv preprint arXiv:2201.09651 
(2022).