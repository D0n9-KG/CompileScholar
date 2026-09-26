Survey of Bias In Text-to-Image Generation: Definition, Evaluation, and Mitigation 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC Zero
 
 
arXiv:2404.01030v3 [cs.CV] 01 May 2024 
 
 

# Survey of Bias In Text-to-Image Generation: Definition, Evaluation, and Mitigation

 
 
 Yixin Wan
 
    
 Arjun Subramonian
 
    
 Anaelia Ovalle
 
    
 Zongyu Lin
 
    
 Ashima Suvarna
 
    
 Christina Chance
 
    
 Hritik Bansal
 
    
 Rebecca Pattichis
 
    
 Kai-Wei Chang
 
 Affiliation:  University of California, Los Angeles
 
 
 Affiliation:  {elaine1wan, arjunsub, anaelia, linzongy21, asuvarna31, cchance, kwchang}@cs.ucla.edu 
 
 Affiliation:  {hbansal, pattichi}@g.ucla.edu 
 

 Abstract 
 
 The recent advancement of large and powerful models with Text-to-Image (T2I) generation abilities—such as OpenAI’s DALLE-3 and Google’s Gemini—enables users to generate high-quality images from textual prompts.
However, it has become increasingly evident that even simple prompts could cause T2I models to exhibit conspicuous social bias in generated images.
Such bias might lead to both allocational and representational harms in society, further marginalizing minority groups.
Noting this problem, a large body of recent works has been dedicated to investigating different dimensions of bias in T2I systems.
However, an extensive review of these studies is lacking, hindering a systematic understanding of current progress and research gaps.
We present the first extensive survey on bias in T2I generative models.
In this survey, we review prior studies on 3 3 dimensions of bias: Gender ,
 Skintone , and Geo-Culture .
Specifically, we discuss how these works define , evaluate , and mitigate different aspects of bias. We found that: (1) while gender and skintone biases are widely studied, geo-cultural bias remains under-explored; (2) most works on gender and skintone bias investigated occupational association, while other aspects are less frequently studied; (3) almost all gender bias works overlook non-binary identities in their studies; (4) evaluation datasets and metrics are scattered, with no unified framework for measuring biases; and (5) current mitigation methods fail to resolve biases comprehensively.
Based on current limitations, we point out future research directions that contribute to human-centric definitions, evaluations, and mitigation of biases.
We hope to highlight the importance of studying biases in T2I systems, as well as encourage future efforts to holistically understand and tackle biases, building fair and trustworthy T2I technologies for everyone.

 
 
 
 
 
 
 | 

 

 
 

## 1 Introduction

 
 Text-To-Image (T2I) models generate accurate images according to textual prompts.
As modern T2I systems such as OpenAI’s DALLE-3  [ 74 ] quickly advance in generation quality and prompt-image alignment, many applications in real-world scenarios have been made possible.
AI-generated images are used in political campaigns  [ 38 ] , films and TV series  [ 23 , 30 ] , games  [ 73 ] , as well as customized advertisements  [ 81 ] .
Some predict that by 2025, 90% of internet content will be AI-generated  [ 36 ] .
However, as our lives are increasingly infiltrated by AI-created visual content, there is an essential question to consider: “What does the world depicted by T2I models look like?” 
Prior works have unveiled severe biases in this depiction  [ 21 , 11 , 33 ] .
For instance, a version of Stable Diffusion  [ 85 ] portrayed the world as being run by white masculine CEOs; dark-skinned men are depicted to be committing crimes, while dark-skinned women are delineated to be flipping burgers  [ 72 ] .
 T2I models’ worldviews are extremely biased, failing to represent the world’s diversity of genders, racial groups, and cultures   [ 60 ] .

 
 
 Observing the problems with bias in T2I models, we raise a second question: “Why are such stereotypes and bias concerning?” 
The answer lies in the social risks they could bring.
Previous works warned that stereotypes and bias could cause representational harms and allocational harms in society  [ 7 , 22 , 12 ] .
For instance, a study showed that Stable Diffusion depicts over 80% of “inmates” with dark skin  [ 72 ] , while people of color only constitute less than half of U.S. prison population  [ 72 , 28 ] .
Such bias might induce false convictions in real world, if the model is applied to help sketch suspected offenders   [ 72 , 79 ] .
Along similar lines,   Wan Chang [104] discovered that T2I models magnify occupational gender bias in society;   Bianchi et al. [9] found that these models default to depicting Western cultures, whilst under-representing others.
Such biases could result in the reinforcement of dominant cultures, propagation or amplification of social stereotypes, under-representation, and even the erasure of socially marginalized communities  [ 9 , 71 , 101 , 96 , 46 , 34 ] .
Furthermore, biases can be propagated by users  [ 68 ] who are unaware of T2I models’ underlying issues and trust their output   [ 31 ] .
Noticing the problem with bias in T2I models, researchers have made efforts to identify different aspects of bias-related risks  [ 11 , 46 ] , as well as develop methods for evaluating and mitigating such issues.
However, little has been done to extensively survey bias definitions and methodologies in previous papers.

 
 
 Towards understanding how prior studies approached bias in T2I models,
we collect 36 36 related papers and review their definitions, evaluation methods, and mitigation strategies for biases.
Details on literature collection are in Appendix A .

 
 
 
 
 
 
 
 Category 
 | 
 
 
 Conceptualization 
 | 
 
 
 Works 
 | 

 
 
 
 
 
 Gender Bias 
 | 
 
 
 Default Generation 
 | 
 
 
 Naik Nushi [71] , Zameshina et al. [111] , Zhang et al. [112] , He et al. [40] , Chinchure et al. [20] , Garcia et al. [35] , Bakr et al. [4] , Esposito et al. [26] 
 | 

 
 | 
 
 
 Occupational Association 
 | 
 
 
 Cho et al. [21] , Bansal et al. [5] , Naik Nushi [71] , Bianchi et al. [9] , Seshadri et al. [92] , Friedrich et al. [33] , Orgad et al. [75] , Kim et al. [47] , Zhang et al. [112] , Chinchure et al. [20] , Shen et al. [94] , Luccioni et al. [60] , Fraser et al. [32] , Vice et al. [102] , Li et al. [54] , Zhang et al. [114] , Wan Chang [104] , Mandal et al. [62] , Wang et al. [105] , Lin et al. [57] , Lee et al. [53] , Sathe et al. [90] , Friedrich et al. [34] 
 | 

 
 | 
 
 
 Characteristics and Interests 
 | 
 
 
 Naik Nushi [71] , Bianchi et al. [9] , Li et al. [54] , Fraser et al. [32] , Zhang et al. [114] , Friedrich et al. [34] , Wang et al. [105] , Mandal et al. [62] , Fraser et al. [31] 
 | 

 
 | 
 
 
 Stereotypical Objects 
 | 
 
 
 Mannering [63] , Bansal et al. [5] , Mandal et al. [62] 
 | 

 
 | 
 
 
 Image Quality 
 | 
 
 
 Naik Nushi [71] , Lee et al. [53] 
 | 

 
 | 
 
 
 Power Dynamics 
 | 
 
 
 Wan Chang [104] 
 | 

 
 | 
 
 
 NSFW Explicit Content 
 | 
 
 
 Ungless et al. [101] , Hao et al. [39] 
 | 

 
 
 
 Skintone Bias 
 | 
 
 
 Default Generation 
 | 
 
 
 Naik Nushi [71] , Zhang et al. [112] , Luccioni et al. [60] , Chinchure et al. [20] , Bakr et al. [4] , Esposito et al. [26] , He et al. [40] 
 | 

 
 | 
 
 
 Occupational Association 
 | 
 
 
 Bansal et al. [5] , Cho et al. [21] , Naik Nushi [71] , Bianchi et al. [9] , Shen et al. [94] , Fraser et al. [32] , Zhang et al. [112] , Lee et al. [53] 
 | 

 
 | 
 
 
 Characteristics and Interests 
 | 
 
 
 Naik Nushi [71] , Fraser et al. [32] , Wang et al. [105] , Fraser et al. [31] , Bianchi et al. [9] 
 | 

 
 
 
 Geo-Cultural Bias 
 | 
 
 
 Geo-Cultural Norms 
 | 
 
 
 Naik Nushi [71] , Bianchi et al. [9] , Liu et al. [59] , Basu et al. [8] 
 | 

 
 | 
 
 
 Characteristics and Interests 
 | 
 
 
 Bianchi et al. [9] , Jha et al. [43] , Struppek et al. [97] 
 | 

 

 Table 1: Collection of papers on biases in T2I models in our study, stratified by bias categories and conceptualizations. A large body of works investigated occupational gender biases, whereas only a few study aspects like power dynamics and geo-cultural characteristics. 
 
 
 We categorize bias definitions along 3 3 primarily studied dimensions in previous works: Gender Presentation Bias , Skintone Bias , and Geo-Cultural Bias .
Through extensive literature review on the 3 3 dimensions of biases, we observe several limitations and research gaps in current studies:
(1) while as many as 32 32 of 36 36 papers explored gender bias, only 6 6 studied geo-cultural biases; (2) among gender bias studies, 23 23 out of 32 32 explored occupational biases, while dimensions like image quality and power dynamics are under-studied; (3) only 6 6 papers studied bias for non-binary genders, while most works overlook gender minority groups; (4) benchmarks and evaluation methods varied from study to study, indicating a lack of unified frameworks for measuring biases; and (5) current mitigation methods fail to efficiently and effectively resolve biases, with the Gemini controversy being an evident example.
Based on these insights, we identify future steps toward human-centric bias definition, evaluation, and mitigation approaches.
This survey provides a broader perspective on “what has been done” and “what can be done” for biases in T2I models, lays out solid knowledge foundations and inspirations for future work, and shed light on potential directions of AI governance.

 
 
 

## 2 Bias Definitions

 
 We observe that prior studies, though all focusing on biases in T2I models, provided different definitions of bias [ 13 ] .
It is important to (1) ensure that bias definitions are grounded in social harms  [ 12 ] , and (2) understand similarities and discrepancies in proposed bias definitions, so as to avoid miscommunication when discussing “bias”.
Based on our literature survey, we identified that previous explorations on bias in T2I models fall into 3 3 major categories: Gender Bias , Skintone Bias , and Geo-Cultural Bias .
In this section, we inductively discuss and summarize how prior works conceptualize these different categories.
Importantly, the conceptualizations in our taxonomy are not mutually-exclusive.

 
 

### 2.1 Gender Bias in T2I Models

 
 T2I models reflect gender stereotypes, such as depicting a “hairdresser” with feminine characteristics and “manager” with masculine characteristics  [ 104 ] .
Since only gender presentation and roles may be perceived from model-synthesized images, the concept of “gender” in these studies refers to perceived gender presentation and roles , not gender or sexual identity.
Additionally, most papers that we surveyed defined “gender” as a binary concept—only 6 6 out of 36 36 papers explored bias issues beyond the binary gender.

 
 
 Bias in Default Generation  
Several prior works defined gender bias as the model’s tendency to portray a particular gender when given gender-neutral prompts (e.g. generate “a person” or “a face”)  [ 71 , 111 , 112 , 26 , 40 ] .
  Chinchure et al. [20] investigated bias in generations for varied prompts from diverse sources.
  Garcia et al. [35] studied bias in images generated from captions describing real pictures.
  Bakr et al. [4] operationalized this bias as the spurious correlations with gender when the prompt is gender-agnostic.

 
 
 Bias in Occupational Association  
Most of the reviewed literature on gender bias in T2I models studied occupational gender bias.
A majority of works in this direction defined bias as the model’s tendency to over-represent or under-represent a particular gender for an occupation  [ 21 , 5 , 71 , 9 , 92 , 33 , 75 , 47 , 112 , 20 , 94 , 60 , 32 , 102 , 54 , 114 , 57 , 53 , 90 , 34 , 104 ] .
  Mandal et al. [62] and   Wang et al. [105] operationalized this bias as the embedding distance between a particular gender and stereotypical occupations.

 
 
 Bias in Characteristics and Interests  
Several previous studies defined bias to be the tendency to generate a particular gender when prompted to depict individuals with certain characteristics or interests like “intelligent”  [ 71 , 9 , 54 , 32 , 114 , 34 ] .
  Wang et al. [105] operationalized this bias as the embedding proximity between a gender and stereotypical interests, such as “science” with men and “art” with women.
  Mandal et al. [62] extended this analysis to interests like “shopping.”
  Fraser et al. [31] briefly explored biased associations between gender and socioeconomic characteristics.

 
 
 Bias in Stereotypical Objects  
  Mannering [63] defined gender bias as a model’s propensity to generate gender-stereotypical objects, such as “ties” for men and “handbags” for women.
  Bansal et al. [5] assess whether models tend to depict a certain gender when prompted to generate people wearing stereotypical items.
  Mandal et al. [62] operationalized this biased association in terms of embedding proximity.

 
 
 Bias in Image Quality  
  Naik Nushi [71] and   Lee et al. [53] coneptualized gender bias in image quality as the tendency to generate higher-quality images (e.g. with better alignment) for a particular gender.

 
 
 Bias in Power Dynamics  
  Wan Chang [104] defined gender bias in power dynamics as the tendency to generate a stereotypical gender for powerful/powerless-indicative prompts, such as men for “CEO” or women for “assistant.”

 
 
 NSFW Explicit Content  
  Ungless et al. [101] defined bias as the tendency to output
Not Safe For Work (NSFW) contents for non-cisgender individuals.   Hao et al. [39] established bias as the disproportionate amplification of input harmful contents for feminine generations.

 
 
 

### 2.2 Skintone bias in T2I Models

 
 T2I models tend to generate social stereotypes related to perceived skin tone.
For example, models are shown to reinforce the “white ideal” by depicting “attractive” individuals as white and “poor” individuals as of color  [ 9 ] .

 
 
 Bias in Default Generation  
Several works conceptualized this aspect of bias as the model’s tendency to generate individuals of a certain skintone when skintone was not explicitly specified in the prompt  [ 71 , 112 , 60 , 4 , 26 , 40 ] .
  Chinchure et al. [20] studied bias in default generation given varied prompts.

 
 
 Bias in Occupational Association  
Related studies mostly defined this dimension of bias to be the model’s tendency to over- or under-represent skin tone groups when depicting certain occupations  [ 5 , 21 , 71 , 9 , 94 , 32 , 112 , 53 ] .

 
 
 Bias in Characteristics and Interests  
Most related works defined this bias as the model’s tendency to depict a specific skin tone when prompted to generate individuals with certain characteristics, such as “poor,” “pleasant,” or “a criminal”  [ 71 , 32 , 105 , 31 , 9 ] .

 
 
 

### 2.3 Geo-Cultural bias in T2I Models

 
 T2I models often generate people and artifacts from over-represented cultures or geographical regions—such as the United States  [ 8 ] .

 
 
 Bias in Geo-Cultural Norms 
  Naik Nushi [71] and   Bianchi et al. [9] defined bias in cultural norms as the tendency to over-represent specific cultures in the default generation setting while under-representing others.
  Liu et al. [59] and   Basu et al. [8] studied the biased norms in depicting non-sensitive words such as clothing and city.

 
 
 Bias in Characteristics and Interests 
  Bianchi et al. [9] conceptualized this bias as the tendency to depict certain cultures with harmful stereotypes —such as portraying “poverty”-indicative images of Africa.
  Jha et al. [43] further studied biased associations with geo-culturally stereotypical facial features.
  Struppek et al. [97] extended the exploration to biased associations with scripts of certain languages.

 
 
 
 

## 3 Bias Evaluation

 
 To understand how previous works measure biases, we review the Evaluation Datasets and Evaluation Metrics in prior studies.
For dataset, we observed a dominance of using different sets of handcrafted prompts, and a lack of unified evaluation benchmarks.
Similarly, we noticed an absence of unified evaluation frameworks and metrics.
In Appendix B Table 2 , we present a table that illustrates the landscape of evaluation metrics in prior works.

 
 

### 3.1 Evaluation Datasets

 
 Manually-Curated Prompts  
Several studies constructed and utilized hand-crafted lists of diagnostic prompts with information about targeted bias aspects, such as occupations and characteristics  [ 39 , 21 , 5 , 71 , 9 , 92 , 62 , 105 , 75 , 31 , 101 , 47 , 111 , 54 , 112 , 94 , 60 , 32 , 63 , 57 , 8 , 26 , 40 , 104 ] .
Additionally, some works created prompts by combining multiple pieces of information, such as gender indicators, human characteristics, and actions  [ 114 , 90 , 102 ] .

 
 
 Proposed Datasets 
  Garcia et al. [35] augmented the Google Conceptual Captions (GCC) dataset  [ 93 ] with age, gender, skintone, and ethnicity annotations.
  Friedrich et al. [34] collected and proposed the Multilingual Assessment of Gender Bias in Image Generation (MAGBIG) dataset with a variety of prompts in 10 10 languages.
  Liu et al. [59] proposed the Cross-Cultural Understanding Benchmark (CCUB) cultural representation dataset for culturally-aware image generation.
  Jha et al. [43] introduced the ViSAGe (Visual Stereotypes Around the Globe) dataset to assess nationality-based stereotypes in T2I models.
  Lee et al. [53] combined self-curated prompts and data from MS-COCO  [ 58 ] and proposed the Holistic Evaluation of Text-to-Image Models (HEIM) benchmark with bias and fairness subcategories.
  Chinchure et al. [20] combined handcrafted prompts and selected ones from the DiffusionDB database  [ 106 ] .
  [ 4 ] combined template-based GPT-3.5-generated prompts [ 76 ] and DiffusionDB to release a benchmark with 39k prompts (9k for bias evaluation).

 
 
 Predefined Datasets 
  Friedrich et al. [33] used a subset of LAION-5B for evaluating occupational gender bias.
 Vice et al. [102] utilized 64 64 prompts from MS-COCO  [ 58 ] .

 
 
 

### 3.2 Evaluation Metrics

 
 We taxonomized bias evaluation metrics into Classification-Based Metrics —where characteristics are directly inferred—and Embedding-Based Metrics , where characteristics might be subtle and measured in latent space.
We observe that a majority of studies adopted classification-based metrics.
However,   Zhang et al. [114] pointed out that (1) identities like gender should not be determined only by appearance, and (2) current classifiers fail to represent transgender individuals.
Future studies can explore other alternatives of attribute classifiers to resolve ethical concerns.

 
 

#### 3.2.1 Classification-Based Metrics

 
 A majority of the literature evaluated bias by classifying demographic characteristics in generated images, such as skintones of depicted individuals.
These works mostly reported bias metrics based on the level of parity in demographic distribution , such as the percentage of men and women, or representation of different cultures, in generated images.

 
 
 Human Annotation  
  Bansal et al. [5] , Naik Nushi [71] ,   Wang et al. [105] ,   Fraser et al. [31] ,   Garcia et al. [35] ,   Fraser et al. [32] , and   Wan Chang [104] utilized human-annotated demographics (e.g. gender, skintone, culture) in generated images for evaluation.
  Garcia et al. [35] reported the percentage difference of unsafe images—as detected by Stable Diffusion v1.4’s Safety Checker module—in real images vs. feminine generations from the images’ captions.
  Liu et al. [59] asked annotators to rank generated images based on best cultural representation and least stereotypical/offensive.
  Basu et al. [8] utilized ratings by human annotators to measure geographical representativeness and realism.
  Zhang et al. [114] utilized human-annotated attribute occurrences in image generations.
  Jha et al. [43] conducted a large-scale annotation task to detect prevalent visual stereotypes in images of 135 identity groups.
  Ungless et al. [101] conducted human evaluation to classify the quality of images generated for non-cisgender people.

 
 
 Classifier-Based Classification  
Several works utilized CLIP’s zero-shot classification abilities to label the demographics of individuals in generated images  [ 5 , 21 , 92 , 75 , 112 , 57 , 53 , 47 ] .
A number of other studies used the pre-trained FairFace classifier  [ 45 ] to annotate characteristics  [ 33 , 34 , 71 ] .
  Shen et al. [94] trained their own gender and race classifiers for evaluation.
  Zhang et al. [112] utilized the Individual Typology Angle (ITA)  [ 18 ] and the method proposed by   Feng et al. [29] to conduct “unbiased” skintone classification.
  Cho et al. [21] used FAN  [ 14 ] , TRUST  [ 29 ] , and ITA to map characteristics of individuals in generated images to their closest skintone in the Monk Skin Tone (MST)  [ 37 ] scale.
  Bakr et al. [4] used ArcFace  [ 24 ] and RetinaFace  [ 25 ] to detect the faces in the generated images, and then used Dex  [ 86 ] to detect facial characteristics related to gender and skintone.
Most of the above studies adopted parity-based metrics  [ 21 , 5 , 92 , 75 , 53 , 47 , 33 , 34 , 71 , 112 , 21 , 4 ] .
Besides parity measurements,   Lin et al. [57] reported how each word in prompts influences biases in generations.
  Seshadri et al. [92] reported percentage differences of feminine images in model generations vs. the training dataset to estimate gender bias amplification.
  Zameshina et al. [111] applied DeepFace  [ 99 ] to detect the gender of individuals in generations, and reported percentage improvement of gender representation after bias mitigation.
  Zhang et al. [114] did not directly classify gender; they trained an attribute classifier to detect elements like “trousers”, and proposed the Gender Presentation Differences (GEP) Score to measure the overall attribute-wise occurrence differences between images generated from gender-explicit vs. gender-neutral prompts.
  Mannering [63] used You Only Look Once (YOLO) v3  [ Redmon2018YOLOv3AI ] for object detection, and listed objects in generations for males and females.
  Hao et al. [39] employed safety classifiers for text and image to measure the amplification of unsafe/harmful contents in generations compared to inputs.

 
 
 VQA-Based Classification  
  Luccioni et al. [60] used BLIP  [ 55 ] , a Visual Question Answering (VQA) model to detect the genders of individuals in generated images.
  Cho et al. [21] , Esposito et al. [26] , Sathe et al. [90] , and   Wan Chang [104] used BLIP-2  [ 56 ] for gender classification, but   Wan Chang [104] found that BLIP-2 fails in gender classification for complicated images, e.g. with multiple people.
Besides works that adopted parity-based metrics  [ 60 , 21 , 26 , 90 ] ,   Wan Chang [104] proposed the Paired Stereotype Test (PST) framework to study biases in multi-person generations, and reported the Stereotype Test Scores (STS) to measure the frequency of occurrence of gender stereotypes in generations.
  Esposito et al. [26] and   Struppek et al. [97] used BLIP-2 for skintone and culture classification, respectively.
  Struppek et al. [97] reported VQA Bias as the percentage increase in culture or skin tone representations when culture-specific characters are added to prompts.
  Vice et al. [102] used BLIP to extract subjects in model output, and reported distribution bias in generated objects, hallucination level, and the generative miss rate (i.e. percentage of outputs that do not align with prompts).
  Chinchure et al. [20] proposed the Text to Image Bias Evaluation Tool (TIBET), which uses the Concept Association Score (CAS) to measure alignment between MiniGPT-v2  [ 19 ] -extracted elements in a generated image vs. in generations from perturbed biased prompts.

 
 
 Embedding Distance-Based Classification  
  Bianchi et al. [9] , Naik Nushi [71] , and   Basu et al. [8] labeled characteristics in generated images by comparing their CLIP  [ 83 ] embedding distances with a set of other images with pre-defined characteristics (e.g. race, gender).
  Lee et al. [53] compared the similarities of skin pixels to MST
scales to classify skintones.
  Li et al. [54] did not specify details, but their visualization illustrated the proximity of images to different genders.
  He et al. [40] used FaceNet [ 91 ] to identify faces, and then matched them to descriptions of gender or skintone using text-image CLIP similarity.
While above works mostly used parity-based metrics  [ 9 , 71 , 53 , 54 ] ,   Basu et al. [8] reported Geographical Representativeness (GR), the average realism rating of generations for different countries.
  Bianchi et al. [9] reported percentage differences of non-white individuals in generated images vs. real-world statistics to estimate the amplification of societal bias.
  Jha et al. [43] identify stereotypes in generations by (1) using CLIP to get top 50 captions for images, and (2) string-match for stereotypes in captions.
They reported the tendency to generate stereotypical attributes compared with random elements, as well as the offensiveness level.
  Vice et al. [102] reported distribution bias in (1) generated objects, (2) hallucination level, and (3) generative miss rate, i.e., percentage of outputs that do not align with prompts.

 
 
 

#### 3.2.2 Embedding-Based Metrics

 
 Association with Biased Characteristics  
  Wang et al. [105] proposed the T2I Association Test (T2IAT).
They mainly reported differential association, which computes differences in the CLIP  [ 83 ] -based proximity of each generation with 2 2 other images generated from prompts with opposite sensitive characteristics.
  Mandal et al. [62] proposed the Multimodal Composite Association Score (MCAS), which is an extension of the Word Embeddings Association Test (WEAT)  [ 16 ] .
MCAS directly examined the mean cosine distance between the CLIP embedding of a text or a generated image of a particular gender and the embeddings of those of a list of stereotypical items.
  Struppek et al. [97] replaced Unicode characters in prompts with seemingly identical non-Latin scripts associated with a specific language, and then calculated Relative Bias as the percentage increase in CLIP-embedding proximity with culturally-explicit texts.
  Luccioni et al. [60] clustered model outputs into 24 different cluster regions constructed with combined gender and ethnicity characteristics, and reported the shares of cluster assignments to infer attributes.

 
 
 Image Quality  
  Naik Nushi [71] computed the Frechet Inception Distance (FID)  [ 41 ] score between model-generated images for each gender and related web images extracted through an online image search API.
  Lee et al. [53] measured image quality bias by calculating changes in the CLIPScore of images generated from prompts with varied gender or dialect characteristics, compared with the original image-caption pair from MS-COCO  [ 58 ] .

 
 
 
 
 

## 4 Bias Mitigation

 
 We categorize bias mitigation strategies into Model Weight Refinement —where models undergo weight adjustments—and Inference-Time and Data Approaches , where model weights remain the same.
Most papers we surveyed adopted prompt-based mitigation approaches, but the method suffers from low robustness and controllability.
In Appendix C Table 3 , we summarize different types of current mitigation approaches.

 
 

### 4.1 Model Weight Refinement

 
 Fine-tuning     Struppek et al. [97] fine-tuned the text encoder to unlearn spurious correlations in T2I generation.
  Esposito et al. [26] trained a T2I model with generations of another model on a diverse set of prompts, with a combination of ethnicity, gender, age, and profession characteristics from LAION-5B and Stereoset  [ 70 ] .
  Liu et al. [59] trained T2I models on negative and positive samples of culturally-representative images with a self-contrastive perceptual loss.
Specifically, they used generations of the pretrained T2I model as negative examples and images from the proposed CCUB dataset as positive ones.

 
 
 Parameter-Efficient Fine-Tuning  
  Li et al. [54] 
inserted “Fair Mapping”, a trainable linear network after the pre-trained text encoder, to map embeddings into a fair space.
During training, they used (1) a fairness loss to diminish associations of bias-sensitive attributes with text embeddings, and (2) a semantic consistency loss to preserve semantic coherence.
  Kim et al. [47] trained a soft prompt prefix using: (1) a de-stereotyping loss that improves diversity in demographic characteristics of generated images, and (2) a regularization loss to ensure faithful representation of the prompt.
  Shen et al. [94] performed Low-rank adaptation of large language models (Lora)  [ 42 ] , fine-tuning with: (1) a distributional alignment loss (DAL) to guide generated characteristics towards a target distribution, and (2) an image semantics-preserving loss.
  Zhang et al. [112] proposed Inclusive Text-to-Image GENeration (ITI-GEN), which appends a set of learnable prompt tokens to prompts to generate reference images with diverse and inclusive demographic characteristics.

 
 
 Model Editing  
  Orgad et al. [75] proposed Text-to-Image Model Editing (TIME), which edits the cross-attention layers of Stable Diffusion by aligning the embedding of a gender-neutral input prompt to the embedding of an anti-stereotype gender-specific prompt.

 
 
 

### 4.2 Inference-Time and Data Approaches

 
 Prompt-Based Mitigation  
Several works used ethical intervention prompts, which instruct models with fairness guidelines, to mitigate biases  [ 5 , 31 , 9 , 104 ] .
However,   Wan Chang [104] found that the fairness intervention approach is not fully controllable, resulting in “overshooting” biases.
  Naik Nushi [71] and   Bianchi et al. [9] utilized prompts with anti-stereotype demographic characteristics, but found that bias persist despite the instructions.
  Friedrich et al. [34] experimented with using gender-neutral prompts for languages with grammatical gender or the “generic masculine”  [ 95 ] ; however, they found that gender-neutral prompts did not remove gender bias, but rather resulted in a degradation of text-to-image alignment and face generation performance.

 
 
 Guided Generation  
  Friedrich et al. [33] proposed Fair Diffusion, a decoding-time method that extends classifier-free guidance to include textual gender characteristics, towards which the model generation is guided.
Fair Diffusion is able to controllably shift bias in any direction based on human-defined fairness instructions.
  He et al. [40] proposed multi-directional guidance paired with iterative distribution alignment, incorporating Kullback–Leibler divergence to guide model output towards a uniform gender distribution.

 
 
 Diverse Sampling  
  Zameshina et al. [111] proposed finding, in an unsupervised manner, vectors in the latent space of Stable Diffusion that are sufficiently distant from each other to decode so that generated images are diverse.

 
 
 Data Augmentation  
  Esposito et al. [26] proposed finetuning T2I models on a diverse set of generated images sourced from different T2I models with varied prompts.

 
 
 
 

## 5 Discussions and Future Directions

 
 Below, we highlight the research gaps present in the study of biases in T2I models, and point out future research directions towards building human-centric approaches for bias definition, evaluation, and mitigation.

 
 

### 5.1 Human-Centric Approaches for Bias Definitions

 
 Clear and Socially-Grounded Conceptualizations of Bias  
Some existing literature failed to articulate a clear definition of bias, not clarifying the specific generation task and associated social stereotypes which were being evaluated or mitigated.
For instance, only a few elaborated on their conceptualization of “gender.”
Since only gender presentation and roles may be perceived from images, it is important to clarify one’s definition, the inherent subjectivity of inferring demographic characteristics, and its implications when discussing research outcomes.
In addition, works studying skin tone and geo-cultural biases often incorrectly conflate skin tone with race or ethnicity and culture with nationality.
Towards precision and transparency, we encourage researchers to provide details on their conceptualization of bias  [ 12 ] .
We further urge researchers to explicitly ground their conceptualizations in concerns about social inequality and power differences  [ 13 , 78 ] .

 
 
 Extension of Bias Dimensions  
Previous studies mostly focused on a limited scope of bias dimensions.
For instance, a large body of works on gender bias explore the unfair associations with occupations.
Meanwhile, only a few research works have explored Geo-Cultural bias in T2I models.
Extending the scope of analysis will provide a more holistic view of the different dimensions of bias (e.g., disability, LGBTQIA+) present in T2I systems.

 
 
 

### 5.2 Human-Centric Approaches for Bias Evaluation

 
 Community Efforts and Positionality  
Socially-grounded evaluations are crucial to AI bias research.
The positionality—subjective experiences, identities, and backgrounds—of researchers and annotators shape their social viewpoints, and might influence their research perspectives   [ 88 ] .
Community efforts seek to understand the influence of technologies on people from different social groups.
Neglecting to examine researcher positionality or solicit human-centric feedback risks missing significant bias considerations, therefore limiting improvements in fairness and inclusivity.
For instance, current works on gender bias depend heavily on binary gender stereotypes.
Consequently, corresponding evaluation strategies are narrowly tailored to address gender binary-specific issues, inadvertently disregarding experiences of individuals beyond the binary, such as transgender and nonbinary communities  [ 77 ] .
Researchers should improve community engagement and clarify their positionalities in bias evaluation to improve transparency and trustworthiness.

 
 
 Trustworthy Automated Evaluation  
Previous works  [ 60 , 21 , 26 , 90 , 104 ] explored the use of VQA models for detecting gendered characteristics of individuals in generations.
However,   Wan Chang [104] pointed out that current VQA models like BLIP-2 fail to correctly identify gender characteristics in complicated images (e.g. with two people), which has become a bottleneck in advancing automated evaluation.
Furthermore, VQA models might carry underlying biases  [ 87 ] .
Building stronger and more reliable automated evaluation methods will allow for scalable approaches to evaluating biases in T2I models.

 
 
 

### 5.3 Human-Centric Approaches for Bias Mitigation

 
 Diverse and Inclusive Mitigation  
Previous works proposed bias mitigation techniques based on their conceptualizations of biases.
However, very few considered user preferences when designing solutions for biases [ 65 ] ;   Ungless et al. [101] was among the few to study how non-binary users prefer to be represented in T2I model generations.
We highlight that diverse outputs do not imply “inclusion” —which refers to the sense of belonging and representation among users  [ 67 ] —and encourage future researchers to consider this point.

 
 
 Robust, Controllable, and Resource-Friendly Mitigation  
Current mitigation methods mostly adopt training and prompt-based approaches.
However, training-based approaches require data and resources, and could result in catastrophic forgetting  [ 61 ] or degraded generation quality  [ 110 ] .
Furthermore, prompt-based approaches suffer from a lack of robustness and controllability  [ 34 , 104 ] , resulting in unstable behaviors such as failing to follow instructions and “over”-mitigating.
Therefore, designing robust, controllable, and resource-friendly mitigation methods is essential for easy and safe T2I applications.

 
 
 Beyond One-size-fits-all Mitigation Approaches One-size-fits-all bias mitigation for T2I models may be at odds with known challenges, such as their tendency to hallucinate, or produce factual content.
For instance, focusing on generating content with equitable distributions across sensitive characteristics may result in inaccurate and offensive image generations, like in the Gemini incident  [ 66 ] .
Among surveyed works, Ungless et al. [101] discussed how prompt blocking—such as curated system prompts, prompt blacklists, or post-prompt moderation—may contribute to the erasure of non-cisnormative identities.

 
 
 One potential solution is aligning generations with diverse human and community preferences   [ 109 , 48 , 108 ] .
For instance, in language modeling, [ 3 , 2 , 100 ] trained safety reward models can guide model generations towards harmless outputs (e.g., refraining to suggest ways to steal the neighbor’s Wi-Fi password).
However, most prior studies on alignment for T2I models [ 52 , 27 , 103 , 98 ] used it to enhance the image quality and image-prompt matching;
limited works have explored alignment as a mitigation strategy for societal biases in T2I models.
However, while we encourage future explorations beyond one-size-fits-all approaches, researchers should critically examine how AI biases intersect with methodological limitations.
As such, proxy objectives, reward hacking, and overoptimization are fundamental challenges to navigate when aligning to human preferences [ 17 , 6 ] .

 
 
 Adaptive and Lifelong Mitigation  
Conceptualizations of bias, bias dimensions, and bias-related stereotypes change over time.
Only mitigating currently-identified biases in models is not sufficient, as new problems might continue to appear in a model’s life cycle.
Instead of static bias mitigation methods, we encourage future studies to explore dynamic, adaptive, and lifelong mitigation strategies that evolve based on continuous feedback and the changing landscape of societal norms and values, as well as community needs.
For example, involving real-time monitoring of model outputs and automatic adjustment of model parameters might help with addressing emerging or varied conceptualizations of bias.

 
 
 
 

## 6 Conclusion

 
 In this paper, we present the first survey on biases in T2I models.
We thoroughly reviewed and summarized the definitions, evaluation metrics, and mitigation methods in 36 36 previous papers on 3 3 dimensions of biases: Gender Presentation , Skintone , and Geo-Cultural .
During our surveying of the papers, we made several meaningful observations: (1) most works focused on studying biases in gender and skintones, whereas very few investigated geo-cultural biases; (2) a majority of gender and skintone bias research explored biased associations with occupations, but few studied aspects like biases in power dynamics and explicit content generation; (3) most works on gender bias do not consider how non-binary individuals are represented; (4) no unified framework for bias evaluation has been consolidated, and bias metrics vary from study to study; and (5) current mitigation methods fail to come up with holistic and effective solutions for biases.
Based on our insights, we highlight future research directions in human-centric approaches for bias definitions, evaluation, and mitigation.
We hope this work offers a helpful overview of the landscape of research on biases in T2I models for researchers and policymakers alike, as well as sheds light on potential paths toward building and regulating trustworthy T2I systems.

 
 
 

## 7 Ethics Statement

 
 This study collects and reviews previous works on biases in T2I models.
Below, we identify a number of important ethical considerations in this paper, as well as in the papers that we survey.

 
 
 Inferring Personal Identities From Images  
Personal identities in this paper and in related works—such as gender, skintone, and geo-culture—are based on their visual presentation in model-generated images.
For instance, prior works often used identity-indicative characteristics as a proxy of the identity itself: using gender characteristics as a proxy for gender, skintone as a proxy for race, and culturally-unique elements as a proxy for culture. This is due to synthesized individuals in generated images not having internal identities.
In particular, we note that: (1) “gender” and “culture” in prior works indicate visual presentations of gender and cultural traits, (2) while previous studies sometimes use “skintone,” “race,” and “ethnicity” interchangeably, “race” and “ethnicity” can never be inferred from images, and (3) some papers incorrectly conflate “culture” and “nationality,” but many countries comprise a multitude of cultures, albeit with some cultures dominant [ 82 ] . Therefore, to ensure clarity of definitions and avoid misconceptions, throughout our survey, we utilized “gender presentation,” “skintone,” and “geo-cultural” to represent dimensions of bias based on visual presentation.

 
 
 Classification Biases  
We underline the limitations of classification strategies in prior works.
When classifying the identities of individuals in generated images, previous studies adopted methods such as human annotation, model-based classifiers, and embedding distance-based classification.
However, both human annotators  [ 80 , 89 , 10 ] and automated methods  [ 49 , 1 , 69 , 15 , 64 , 107 , 51 , 84 , 50 , 35 ] have been shown to possess bias.
Therefore, using these classification tools will inevitably risk propagating their bias in T2I bias evaluation results.
We stress that researchers be aware of this limitation, and transparently and critically discuss their approaches to classification.

 
 
 AI For Justice-Centered Applications  
The definitions and methodologies in this survey should not support evaluations and mitigation for bias in unjust applications of generative AI.
We emphasize that approaches for measuring or resolving biases should not be exploited to ethics-wash applications of generative AI that enable surveillance, harming or suppressing workers  [ 44 ] , weaponization, etc.

 
 
 

## References

 
 
 [1] 
 
Salem Hamed Abdurrahim, Salina Abdul Samad, and Aqilah Baseri Huddin.

 
 Review on the effects of age, gender, and race demographics on automatic face recognition.

 
 The Visual Computer , 34:1617–1630, 2018.

 

 
 [2] 
 
Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al.

 
 Training a helpful and harmless assistant with reinforcement learning from human feedback.

 
 arXiv preprint arXiv:2204.05862 , 2022a.

 

 
 [3] 
 
Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al.

 
 Constitutional ai: Harmlessness from ai feedback.

 
 arXiv preprint arXiv:2212.08073 , 2022b.

 

 
 [4] 
 
Eslam Mohamed Bakr, Pengzhan Sun, Xiaogian Shen, Faizan Farooq Khan, Li Erran Li, and Mohamed Elhoseiny.

 
 Hrs-bench: Holistic, reliable and scalable benchmark for text-to-image models.

 
 In Proceedings of the IEEE/CVF International Conference on Computer Vision , pp. 20041–20053, 2023.

 

 
 [5] 
 
Hritik Bansal, Da Yin, Masoud Monajatipoor, and Kai-Wei Chang.

 
 How well can text-to-image generative models understand ethical natural language interventions?

 
 arXiv preprint arXiv:2210.15230 , 2022.

 

 
 [6] 
 
Hritik Bansal, John Dang, and Aditya Grover.

 
 Peering through preferences: Unraveling feedback acquisition for aligning large language models.

 
 arXiv preprint arXiv:2308.15812 , 2023.

 

 
 [7] 
 
Solon Barocas, Kate Crawford, Aaron Shapiro, and Hanna Wallach.

 
 The problem with bias: Allocative versus representational harms in machine learning.

 
 In 9th Annual conference of the special interest group for computing, information and society , pp.  1. Philadelphia, PA, USA, 2017.

 

 
 [8] 
 
Abhipsa Basu, R Venkatesh Babu, and Danish Pruthi.

 
 Inspecting the geographical representativeness of images from text-to-image models.

 
 In Proceedings of the IEEE/CVF International Conference on Computer Vision , pp. 5136–5147, 2023.

 

 
 [9] 
 
Federico Bianchi, Pratyusha Kalluri, Esin Durmus, Faisal Ladhak, Mira Cheng, Debora Nozza, Tatsunori Hashimoto, Dan Jurafsky, James Zou, Aylin Caliskan, et al.

 
 Easily accessible text-to-image generation amplifies demographic stereotypes at large scale.

 
 In FAccT’23: Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency . Association for Computing Machinery, 2023.

 

 
 [10] 
 
Laura Biester, Vanita Sharma, Ashkan Kazemi, Naihao Deng, Steven Wilson, and Rada Mihalcea.

 
 Analyzing the effects of annotator gender across NLP tasks.

 
 In Gavin Abercrombie, Valerio Basile, Sara Tonelli, Verena Rieser, and Alexandra Uma (eds.), Proceedings of the 1st Workshop on Perspectivist Approaches to NLP @LREC2022 , pp. 10–19, Marseille, France, June 2022. European Language Resources Association.

 
 URL https://aclanthology.org/2022.nlperspectives-1.2 .

 

 
 [11] 
 
Charlotte Bird, Eddie Ungless, and Atoosa Kasirzadeh.

 
 Typology of risks of generative text-to-image models.

 
 In Proceedings of the 2023 AAAI/ACM Conference on AI, Ethics, and Society , pp. 396–410, 2023.

 

 
 [12] 
 
Su Lin Blodgett, Solon Barocas, Hal Daumé III, and Hanna Wallach.

 
 Language (technology) is power: A critical survey of “bias” in NLP.

 
 In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (eds.), Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , pp. 5454–5476, Online, July 2020. Association for Computational Linguistics.

 
 doi: 10.18653/v1/2020.acl-main.485 .

 
 URL https://aclanthology.org/2020.acl-main.485 .

 

 
 [13] 
 
Su Lin Blodgett, Gilsinia Lopez, Alexandra Olteanu, Robert Sim, and Hanna Wallach.

 
 Stereotyping norwegian salmon: An inventory of pitfalls in fairness benchmark datasets.

 
 In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) , pp. 1004–1015, 2021.

 

 
 [14] 
 
Adrian Bulat and Georgios Tzimiropoulos.

 
 How far are we from solving the 2d 3d face alignment problem?(and a dataset of 230,000 3d facial landmarks).

 
 In Proceedings of the IEEE international conference on computer vision , pp. 1021–1030, 2017.

 

 
 [15] 
 
Joy Buolamwini and Timnit Gebru.

 
 Gender shades: Intersectional accuracy disparities in commercial gender classification.

 
 In FAT , 2018.

 
 URL https://api.semanticscholar.org/CorpusID:3298854 .

 

 
 [16] 
 
Aylin Caliskan, Joanna J Bryson, and Arvind Narayanan.

 
 Semantics derived automatically from language corpora contain human-like biases.

 
 Science , 356(6334):183–186, 2017.

 

 
 [17] 
 
Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, et al.

 
 Open problems and fundamental limitations of reinforcement learning from human feedback.

 
 arXiv preprint arXiv:2307.15217 , 2023.

 

 
 [18] 
 
Alain Chardon, Isabelle Cretois, and Colette Hourseau.

 
 Skin colour typology and suntanning pathways.

 
 International Journal of Cosmetic Science , 13, 1991.

 
 URL https://api.semanticscholar.org/CorpusID:25650931 .

 

 
 [19] 
 
Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny.

 
 Minigpt-v2: large language model as a unified interface for vision-language multi-task learning, 2023.

 

 
 [20] 
 
Aditya Chinchure, Pushkar Shukla, Gaurav Bhatt, Kiri Salij, Kartik Hosanagar, Leonid Sigal, and Matthew Turk.

 
 Tibet: Identifying and evaluating biases in text-to-image generative models.

 
 arXiv preprint arXiv:2312.01261 , 2023.

 

 
 [21] 
 
Jaemin Cho, Abhay Zala, and Mohit Bansal.

 
 Dall-eval: Probing the reasoning skills and social biases of text-to-image generation models, 2023.

 

 
 [22] 
 
Kate Crawford.

 
 The trouble with bias.

 
 Keynote at NeurIPS, 2017.

 
 URL https://www.youtube.com/watch?v=fMym_BKWQzk .

 

 
 [23] 
 
Tom Davenport.

 
 Cuebric:generative ai comes to hollywood.

 
 2023.

 
 URL https://www.forbes.com/sites/tomdavenport/2023/03/13/cuebric-generative-ai-comes-to-hollywood/?sh=19b07abb174b .

 

 
 [24] 
 
Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou.

 
 Arcface: Additive angular margin loss for deep face recognition.

 
 In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pp. 4690–4699, 2019.

 

 
 [25] 
 
Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou.

 
 Retinaface: Single-shot multi-level face localisation in the wild.

 
 In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pp. 5203–5212, 2020.

 

 
 [26] 
 
Piero Esposito, Parmida Atighehchian, Anastasis Germanidis, and Deepti Ghadiyaram.

 
 Mitigating stereotypical biases in text to image generative systems.

 
 arXiv preprint arXiv:2310.06904 , 2023.

 

 
 [27] 
 
Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee.

 
 Reinforcement learning for fine-tuning text-to-image diffusion models.

 
 Advances in Neural Information Processing Systems , 36, 2024.

 

 
 [28] 
 
Federal Bureau of Prisons.

 
 BOP Statistics: Inmate Race — bop.gov.

 
 https://www.bop.gov/about/statistics/statistics_inmate_race.jsp .

 

 
 [29] 
 
Haiwen Feng, Timo Bolkart, Joachim Tesch, Michael J Black, and Victoria Abrevaya.

 
 Towards racially unbiased skin tone estimation via scene disambiguation.

 
 In European Conference on Computer Vision , pp. 72–90. Springer, 2022.

 

 
 [30] 
 
Charlie Fink.

 
 Vr film producer announces ai film.

 
 2023.

 
 URL https://www.forbes.com/sites/charliefink/2023/03/02/vr-film-producer-announces-ai-film/?sh=553011426ab9 .

 

 
 [31] 
 
Kathleen C. Fraser, Svetlana Kiritchenko, and Isar Nejadgholi.

 
 Diversity is not a one-way street: Pilot study on ethical interventions for racial bias in text-to-image systems.

 
 In 14th International Conference on Computational Creativity (ICCC) . Waterloo, ON, Canada, 2023a.

 

 
 [32] 
 
Kathleen C Fraser, Svetlana Kiritchenko, and Isar Nejadgholi.

 
 A friendly face: Do text-to-image systems rely on stereotypes when the input is under-specified?

 
 arXiv preprint arXiv:2302.07159 , 2023b.

 

 
 [33] 
 
Felix Friedrich, Manuel Brack, Lukas Struppek, Dominik Hintersdorf, Patrick Schramowski, Sasha Luccioni, and Kristian Kersting.

 
 Fair diffusion: Instructing text-to-image generation models on fairness, 2023.

 

 
 [34] 
 
Felix Friedrich, Katharina Hämmerl, Patrick Schramowski, Jindrich Libovicky, Kristian Kersting, and Alexander Fraser.

 
 Multilingual text-to-image generation magnifies gender stereotypes and prompt engineering may not help you.

 
 arXiv e-prints , pp. arXiv–2401, 2024.

 

 
 [35] 
 
Noa Garcia, Yusuke Hirota, Yankun Wu, and Yuta Nakashima.

 
 Uncurated image-text datasets: Shedding light on demographic bias.

 
 In CVPR , 2023.

 

 
 [36] 
 
Alexandra Garfinkle.

 
 90
 
 2023.

 
 URL https://finance.yahoo.com/news/90-of-online-content-could-be-generated-by-ai-by-2025-expert-says-201023872.html .

 

 
 [37] 
 
Google Responsible AI.

 
 Improving skin tone evaluation in machine learning.

 
 2022.

 
 URL https://skintone.google/ .

 

 
 [38] 
 
GOP.

 
 Beat Biden — youtube.com.

 
 https://www.youtube.com/watch?v=kLMMxgtxQ1Y t=32s , 2023.

 

 
 [39] 
 
Susan Hao, Renee Shelby, Yuchi Liu, Hansa Srinivasan, Mukul Bhutani, Burcu Karagol Ayan, Shivani Poddar, and Sarah Laszlo.

 
 Harm amplification in text-to-image models.

 
 arXiv preprint arXiv:2402.01787 , 2024.

 

 
 [40] 
 
Ruifei He, Chuhui Xue, Haoru Tan, Wenqing Zhang, Yingchen Yu, Song Bai, and Xiaojuan Qi.

 
 Debiasing text-to-image diffusion models, 2024.

 

 
 [41] 
 
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter.

 
 Gans trained by a two time-scale update rule converge to a local nash equilibrium.

 
 Advances in neural information processing systems , 30, 2017.

 

 
 [42] 
 
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.

 
 LoRA: Low-rank adaptation of large language models.

 
 In International Conference on Learning Representations , 2022.

 
 URL https://openreview.net/forum?id=nZeVKeeFYf9 .

 

 
 [43] 
 
Akshita Jha, Vinodkumar Prabhakaran, Remi Denton, Sarah Laszlo, Shachi Dave, Rida Qadri, Chandan K. Reddy, and Sunipa Dev.

 
 Beyond the surface: A global-scale analysis of visual stereotypes in text-to-image generation.

 
 ArXiv , abs/2401.06310, 2024.

 
 URL https://api.semanticscholar.org/CorpusID:267959832 .

 

 
 [44] 
 
Harry H. Jiang, Lauren Brown, Jessica Cheng, Mehtab Khan, Abhishek Gupta, Deja Workman, Alex Hanna, Johnathan Flowers, and Timnit Gebru.

 
 Ai art and its impact on artists.

 
 In Proceedings of the 2023 AAAI/ACM Conference on AI, Ethics, and Society , AIES ’23, pp. 363–374, New York, NY, USA, 2023. Association for Computing Machinery.

 
 ISBN 9798400702310.

 
 doi: 10.1145/3600211.3604681 .

 
 URL https://doi.org/10.1145/3600211.3604681 .

 

 
 [45] 
 
Kimmo Kärkkäinen and Jungseock Joo.

 
 Fairface: Face attribute dataset for balanced race, gender, and age.

 
 arXiv preprint arXiv:1908.04913 , 2019.

 

 
 [46] 
 
Amelia Katirai, Noa Garcia, Kazuki Ide, Yuta Nakashima, and Atsuo Kishimoto.

 
 Situating the social issues of image generation models in the model life cycle: a sociotechnical approach.

 
 arXiv preprint arXiv:2311.18345 , 2023.

 

 
 [47] 
 
Eunji Kim, Siwon Kim, Chaehun Shin, and Sungroh Yoon.

 
 De-stereotyping Text-to-image Models through Prompt Tuning.

 
 https://openreview.net/forum?id=yNyywJln2R , 2023.

 

 
 [48] 
 
Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy.

 
 Pick-a-pic: An open dataset of user preferences for text-to-image generation.

 
 Advances in Neural Information Processing Systems , 36, 2024.

 

 
 [49] 
 
Brendan F. Klare, Mark J. Burge, Joshua C. Klontz, Richard W. Vorder Bruegge, and Anil K. Jain.

 
 Face recognition performance: Role of demographic information.

 
 IEEE Transactions on Information Forensics and Security , 7(6):1789–1801, 2012.

 
 doi: 10.1109/TIFS.2012.2214212 .

 

 
 [50] 
 
Anoop Krishnan and Ajita Rattani.

 
 A novel approach for bias mitigation of gender classification algorithms using consistency regularization.

 
 Image and Vision Computing , 137:104793, 2023.

 
 ISSN 0262-8856.

 
 doi: https://doi.org/10.1016/j.imavis.2023.104793 .

 
 URL https://www.sciencedirect.com/science/article/pii/S0262885623001671 .

 

 
 [51] 
 
Anoop Krishnan, Ali Almadan, and Ajita Rattani.

 
 Understanding fairness of gender classification algorithms across gender-race groups.

 
 In 2020 19th IEEE international conference on machine learning and applications (ICMLA) , pp. 1028–1035. IEEE, 2020.

 

 
 [52] 
 
Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu.

 
 Aligning text-to-image models using human feedback.

 
 arXiv preprint arXiv:2302.12192 , 2023.

 

 
 [53] 
 
Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Teufel, Marco Bellagente, et al.

 
 Holistic evaluation of text-to-image models.

 
 Advances in Neural Information Processing Systems , 36, 2024.

 

 
 [54] 
 
Jia Li, Lijie Hu, Jingfeng Zhang, Tianhang Zheng, Hua Zhang, and Di Wang.

 
 Fair text-to-image diffusion via fair mapping.

 
 arXiv preprint arXiv:2311.17695 , 2023a.

 

 
 [55] 
 
Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi.

 
 Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation.

 
 In International conference on machine learning , pp. 12888–12900. PMLR, 2022.

 

 
 [56] 
 
Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.

 
 Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023b.

 

 
 [57] 
 
Alexander Lin, Lucas Monteiro Paes, Sree Harsha Tanneru, Suraj Srinivas, and Himabindu Lakkaraju.

 
 Word-level explanations for analyzing bias in text-to-image models, 2023.

 

 
 [58] 
 
Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick.

 
 Microsoft coco: Common objects in context.

 
 In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13 , pp. 740–755. Springer, 2014.

 

 
 [59] 
 
Zhixuan Liu, Peter Schaldenbrand, Beverley-Claire Okogwu, Wenxuan Peng, Youngsik Yun, Andrew Hundt, Jihie Kim, and Jean Oh.

 
 Scoft: Self-contrastive fine-tuning for equitable image generation.

 
 arXiv preprint arXiv:2401.08053 , 2024.

 

 
 [60] 
 
Alexandra Sasha Luccioni, Christopher Akiki, Margaret Mitchell, and Yacine Jernite.

 
 Stable bias: Analyzing societal representations in diffusion models.

 
 arXiv preprint arXiv:2303.11408 , 2023.

 

 
 [61] 
 
Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang.

 
 An empirical study of catastrophic forgetting in large language models during continual fine-tuning.

 
 arXiv preprint arXiv:2308.08747 , 2023.

 

 
 [62] 
 
Abhishek Mandal, Susan Leavy, and Suzanne Little.

 
 Multimodal composite association score: Measuring gender bias in generative multimodal models, 2023.

 

 
 [63] 
 
Harvey Mannering.

 
 Analysing gender bias in text-to-image models using object detection.

 
 arXiv preprint arXiv:2307.08025 , 2023.

 

 
 [64] 
 
Daniel McDuff, Shuang Ma, Yale Song, and Ashish Kapoor.

 
 Characterizing bias in classifiers using generative models.

 
 Advances in neural information processing systems , 32, 2019.

 

 
 [65] 
 
Ninareh Mehrabi, Palash Goyal, Apurv Verma, Jwala Dhamala, Varun Kumar, Qian Hu, Kai-Wei Chang, Richard Zemel, Aram Galstyan, and Rahul Gupta.

 
 Resolving ambiguities in text-to-image generative models.

 
 In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pp. 14367–14388, 2023.

 

 
 [66] 
 
Margaret Mitchell.

 
 Ethical AI Isn’t to Blame for Google’s Gemini Debacle — time.com.

 
 https://time.com/6836153/ethical-ai-google-gemini-debacle/ , 2024.

 

 
 [67] 
 
Margaret Mitchell, Dylan Baker, Nyalleng Moorosi, Emily Denton, Ben Hutchinson, Alex Hanna, Timnit Gebru, and Jamie Morgenstern.

 
 Diversity and inclusion metrics in subset selection.

 
 In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society , pp. 117–123, 2020.

 

 
 [68] 
 
Luke Munn, Liam Magee, and Vanicka Arora.

 
 Unmaking ai imagemaking: A methodological toolkit for critical investigation, 2023.

 

 
 [69] 
 
Vidya Muthukumar, Tejaswini Pedapati, Nalini Ratha, Prasanna Sattigeri, Chai-Wah Wu, Brian Kingsbury, Abhishek Kumar, Samuel Thomas, Aleksandra Mojsilovic, and Kush R. Varshney.

 
 Understanding unequal gender classification accuracy from face images, 2018.

 

 
 [70] 
 
Moin Nadeem, Anna Bethke, and Siva Reddy.

 
 StereoSet: Measuring stereotypical bias in pretrained language models.

 
 In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) , pp. 5356–5371, Online, August 2021. Association for Computational Linguistics.

 
 doi: 10.18653/v1/2021.acl-long.416 .

 
 URL https://aclanthology.org/2021.acl-long.416 .

 

 
 [71] 
 
Ranjita Naik and Besmira Nushi.

 
 Social biases through the text-to-image generation lens, 2023.

 

 
 [72] 
 
Leonardo Nicoletti and Dina Bass.

 
 Humans are biased. generative ai is even worse.

 
 2023.

 
 URL https://www.bloomberg.com/graphics/2023-generative-ai-bias/ .

 

 
 [73] 
 
Evgeny Obedkov.

 
 How ai-assisted rpg tales of syn utilizes stable diffusion and chatgpt to create assets and dialogues.

 
 2023.

 
 URL {https://gameworldobserver.com/2023/03/06/tales-of-syn-ai-rpg-stable-diffusion-chatgpt-game} .

 

 
 [74] 
 
OpenAI.

 
 Dall·e 3 system card, Oct 2023.

 
 URL https://openai.com/research/dall-e-3-system-card .

 

 
 [75] 
 
Hadas Orgad, Bahjat Kawar, and Yonatan Belinkov.

 
 Editing implicit assumptions in text-to-image diffusion models.

 
 2023 IEEE/CVF International Conference on Computer Vision (ICCV) , pp. 7030–7038, 2023.

 
 URL https://api.semanticscholar.org/CorpusID:257505246 .

 

 
 [76] 
 
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.

 
 Training language models to follow instructions with human feedback.

 
 Advances in Neural Information Processing Systems , 35:27730–27744, 2022.

 

 
 [77] 
 
Anaelia Ovalle, Palash Goyal, Jwala Dhamala, Zachary Jaggers, Kai-Wei Chang, Aram Galstyan, Richard Zemel, and Rahul Gupta.

 
 “i’m fully who i am”: Towards centering transgender and non-binary voices to measure biases in open language generation.

 
 In Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency , pp. 1246–1266, 2023a.

 

 
 [78] 
 
Anaelia Ovalle, Arjun Subramonian, Vagrant Gautam, Gilbert Gee, and Kai-Wei Chang.

 
 Factoring the matrix of domination: A critical review and reimagination of intersectionality in ai fairness.

 
 In Proceedings of the 2023 AAAI/ACM Conference on AI, Ethics, and Society , AIES ’23, pp. 496–511, New York, NY, USA, 2023b. Association for Computing Machinery.

 
 ISBN 9798400702310.

 
 doi: 10.1145/3600211.3604705 .

 
 URL https://doi.org/10.1145/3600211.3604705 .

 

 
 [79] 
 
Basudha Pal, Arunkumar Kannan, Ram Prabhakar Kathirvel, Alice J. O’Toole, and Rama Chellappa.

 
 Gaussian harmony: Attaining fairness in diffusion-based face generation models, 2023.

 

 
 [80] 
 
Rahul Pandey, Carlos Castillo, and Hemant Purohit.

 
 Modeling human annotation errors to design bias-aware systems for social stream processing.

 
 In Proceedings of the 2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining , pp. 374–377, 2019.

 

 
 [81] 
 
Danny Postma.

 
 AI Modelling Agency — Deep Agency — deepagency.com.

 
 https://www.deepagency.com/ .

 

 
 [82] 
 
Rida Qadri, Renee Shelby, Cynthia L. Bennett, and Emily Denton.

 
 Ai’s regimes of representation: A community-centered study of text-to-image models in south asia.

 
 In Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency , FAccT ’23, pp. 506–517, New York, NY, USA, 2023. Association for Computing Machinery.

 
 ISBN 9798400701924.

 
 doi: 10.1145/3593013.3594016 .

 
 URL https://doi.org/10.1145/3593013.3594016 .

 

 
 [83] 
 
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever.

 
 Learning transferable visual models from natural language supervision.

 
 In International Conference on Machine Learning , 2021.

 
 URL https://api.semanticscholar.org/CorpusID:231591445 .

 

 
 [84] 
 
Sreeraj Ramachandran and Ajita Rattani.

 
 Deep generative views to mitigate gender classification bias across gender-race groups.

 
 In International Conference on Pattern Recognition , pp. 551–569. Springer, 2022.

 

 
 [85] 
 
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

 
 High-resolution image synthesis with latent diffusion models, 2021.

 

 
 [86] 
 
Rasmus Rothe, Radu Timofte, and Luc Van Gool.

 
 Dex: Deep expectation of apparent age from a single image.

 
 In Proceedings of the IEEE international conference on computer vision workshops , pp. 10–15, 2015.

 

 
 [87] 
 
Gabriele Ruggeri and Debora Nozza.

 
 A multi-dimensional study on bias in vision-language models.

 
 In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Findings of the Association for Computational Linguistics: ACL 2023 , pp. 6445–6455, Toronto, Canada, July 2023. Association for Computational Linguistics.

 
 doi: 10.18653/v1/2023.findings-acl.403 .

 
 URL https://aclanthology.org/2023.findings-acl.403 .

 

 
 [88] 
 
Sebastin Santy, Jenny T Liang, Ronan Le Bras, Katharina Reinecke, and Maarten Sap.

 
 Nlpositionality: Characterizing design biases of datasets and models.

 
 arXiv preprint arXiv:2306.01943 , 2023.

 

 
 [89] 
 
Maarten Sap, Swabha Swayamdipta, Laura Vianna, Xuhui Zhou, Yejin Choi, and Noah A. Smith.

 
 Annotators with attitudes: How annotator beliefs and identities bias toxic language detection.

 
 In Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz (eds.), Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pp. 5884–5906, Seattle, United States, July 2022. Association for Computational Linguistics.

 
 doi: 10.18653/v1/2022.naacl-main.431 .

 
 URL https://aclanthology.org/2022.naacl-main.431 .

 

 
 [90] 
 
Ashutosh Sathe, Prachi Jain, and Sunayana Sitaram.

 
 A unified framework and dataset for assessing gender bias in vision-language models, 2024.

 

 
 [91] 
 
Florian Schroff, Dmitry Kalenichenko, and James Philbin.

 
 Facenet: A unified embedding for face recognition and clustering.

 
 In Proceedings of the IEEE conference on computer vision and pattern recognition , pp. 815–823, 2015.

 

 
 [92] 
 
Preethi Seshadri, Sameer Singh, and Yanai Elazar.

 
 The bias amplification paradox in text-to-image generation, 2023.

 

 
 [93] 
 
Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut.

 
 Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning.

 
 In Iryna Gurevych and Yusuke Miyao (eds.), Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pp. 2556–2565, Melbourne, Australia, July 2018. Association for Computational Linguistics.

 
 doi: 10.18653/v1/P18-1238 .

 
 URL https://aclanthology.org/P18-1238 .

 

 
 [94] 
 
Xudong Shen, Chao Du, Tianyu Pang, Min Lin, Yongkang Wong, and Mohan Kankanhalli.

 
 Finetuning text-to-image diffusion models for fairness.

 
 arXiv preprint arXiv:2311.07604 , 2023.

 

 
 [95] 
 
Jeanette Silveira.

 
 Generic masculine words and thinking.

 
 Women’s Studies International Quarterly , 3(2):165–178, 1980.

 
 ISSN 0148-0685.

 
 doi: https://doi.org/10.1016/S0148-0685(80)92113-2 .

 
 URL https://www.sciencedirect.com/science/article/pii/S0148068580921132 .

 
 The voices and words of women and men.

 

 
 [96] 
 
Irene Solaiman, Zeerak Talat, William Agnew, Lama Ahmad, Dylan Baker, Su Lin Blodgett, Hal Daumé III au2, Jesse Dodge, Ellie Evans, Sara Hooker, Yacine Jernite, Alexandra Sasha Luccioni, Alberto Lusoli, Margaret Mitchell, Jessica Newman, Marie-Therese Png, Andrew Strait, and Apostol Vassilev.

 
 Evaluating the social impact of generative ai systems in systems and society, 2023.

 

 
 [97] 
 
Lukas Struppek, Dom Hintersdorf, Felix Friedrich, Patrick Schramowski, Kristian Kersting, et al.

 
 Exploiting cultural biases via homoglyphs in text-to-image synthesis.

 
 Journal of Artificial Intelligence Research , 78:1017–1068, 2023.

 

 
 [98] 
 
Jiao Sun, Deqing Fu, Yushi Hu, Su Wang, Royi Rassin, Da-Cheng Juan, Dana Alon, Charles Herrmann, Sjoerd van Steenkiste, Ranjay Krishna, et al.

 
 Dreamsync: Aligning text-to-image generation with image understanding feedback.

 
 arXiv preprint arXiv:2311.17946 , 2023.

 

 
 [99] 
 
Yaniv Taigman, Ming Yang, Marc’Aurelio Ranzato, and Lior Wolf.

 
 Deepface: Closing the gap to human-level performance in face verification.

 
 In 2014 IEEE Conference on Computer Vision and Pattern Recognition , pp. 1701–1708, 2014.

 
 doi: 10.1109/CVPR.2014.220 .

 

 
 [100] 
 
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.

 
 Llama 2: Open foundation and fine-tuned chat models.

 
 arXiv preprint arXiv:2307.09288 , 2023.

 

 
 [101] 
 
Eddie Ungless, Bjorn Ross, and Anne Lauscher.

 
 Stereotypes and smut: The (mis)representation of non-cisgender identities by text-to-image models.

 
 In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Findings of the Association for Computational Linguistics: ACL 2023 , pp. 7919–7942, Toronto, Canada, July 2023. Association for Computational Linguistics.

 
 doi: 10.18653/v1/2023.findings-acl.502 .

 
 URL https://aclanthology.org/2023.findings-acl.502 .

 

 
 [102] 
 
Jordan Vice, Naveed Akhtar, Richard Hartley, and Ajmal Mian.

 
 Quantifying bias in text-to-image generative models, 2023.

 

 
 [103] 
 
Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik.

 
 Diffusion model alignment using direct preference optimization.

 
 arXiv preprint arXiv:2311.12908 , 2023.

 

 
 [104] 
 
Yixin Wan and Kai-Wei Chang.

 
 The male ceo and the female assistant: Probing gender biases in text-to-image models through paired stereotype test.

 
 arXiv preprint arXiv:2402.11089 , 2024.

 

 
 [105] 
 
Jialu Wang, Xinyue Liu, Zonglin Di, Yang Liu, and Xin Wang.

 
 T2IAT: Measuring valence and stereotypical biases in text-to-image generation.

 
 In Findings of the Association for Computational Linguistics: ACL 2023 , pp. 2560–2574, Toronto, Canada, July 2023a. Association for Computational Linguistics.

 
 doi: 10.18653/v1/2023.findings-acl.160 .

 
 URL https://aclanthology.org/2023.findings-acl.160 .

 

 
 [106] 
 
Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau.

 
 DiffusionDB: A large-scale prompt gallery dataset for text-to-image generative models.

 
 In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pp. 893–911, Toronto, Canada, July 2023b. Association for Computational Linguistics.

 
 doi: 10.18653/v1/2023.acl-long.51 .

 
 URL https://aclanthology.org/2023.acl-long.51 .

 

 
 [107] 
 
Wenying Wu, Pavlos Protopapas, Zheng Yang, and Panagiotis Michalatos.

 
 Gender classification and bias mitigation in facial images.

 
 In Proceedings of the 12th ACM Conference on Web Science , pp. 106–114, 2020.

 

 
 [108] 
 
Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li.

 
 Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis.

 
 arXiv preprint arXiv:2306.09341 , 2023.

 

 
 [109] 
 
Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong.

 
 Imagereward: Learning and evaluating human preferences for text-to-image generation.

 
 Advances in Neural Information Processing Systems , 36, 2024.

 

 
 [110] 
 
Zonghan Yang, Xiaoyuan Yi, Peng Li, Yang Liu, and Xing Xie.

 
 Unified detoxifying and debiasing in language generation via inference-time adaptive optimization.

 
 arXiv preprint arXiv:2210.04492 , 2022.

 

 
 [111] 
 
Mariia Zameshina, Olivier Teytaud, and Laurent Najman.

 
 Diverse diffusion: Enhancing image diversity in text-to-image generation, 2023.

 

 
 [112] 
 
Cheng Zhang, Xuanbai Chen, Siqi Chai, Henry Chen Wu, Dmitry Lagun, Thabo Beeler, and Fernando De la Torre.

 
 ITI-GEN: Inclusive text-to-image generation.

 
 In ICCV , 2023a.

 

 
 [113] 
 
R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang.

 
 The unreasonable effectiveness of deep features as a perceptual metric.

 
 In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pp. 586–595, Los Alamitos, CA, USA, jun 2018. IEEE Computer Society.

 
 doi: 10.1109/CVPR.2018.00068 .

 
 URL https://doi.ieeecomputersociety.org/10.1109/CVPR.2018.00068 .

 

 
 [114] 
 
Yanzhe Zhang, Lu Jiang, Greg Turk, and Diyi Yang.

 
 Auditing gender presentation differences in text-to-image models, 2023b.

 

 
 
 
 

## Appendix A Literature Selection Methodology

 
 To select the papers for our analysis, we began with an examination of the most-cited papers on the topic of “bias” and “fairness” for Text-to-Image (T2I) models, as indexed by Google Scholar and arXiv.
As the field is relatively new with a large body of very recent papers, we include both peer-reviewed works and preprints.
This initial step provided a foundation from which we explored further, delving into works cited by and that cited these papers.
This approach was designed to ensure an extensive and informed review of the field, capturing a wide range of perspectives and methodologies.
While we endeavored to be thorough in the literature selection process, we acknowledge the possibility that a small portion of relevant papers may have been inadvertently overlooked.
We recognize the inherent limitations in any literature review or collection process.

 
 
 

## Appendix B Summarization of Evaluation Metrics

 
 Table 2 summarizes and stratifies evaluation methods in previous works.
 18 18 out of the total of 36 36 works that we studied adopted classifier-based approaches to identify demographic characteristics in generated images.
Only a few explored embedding-based measurements such as generation diversity and image quality.

 
 
 
 
 
 
 
 Metric Type 
 | 
 
 
 Category 
 | 
 
 
 Works 
 | 

 
 
 
 
 
 Classification-Based Metrics 
 | 
 
 
 Human Annotation 
 | 
 
 
 Bansal et al. [5] , Naik Nushi [71] , Wang et al. [105] , Fraser et al. [31] , Garcia et al. [35] , Fraser et al. [32] , Basu et al. [8] , Zhang et al. [114] , Ungless et al. [101] , Wan Chang [104] , Liu et al. [59] , Jha et al. [43] 
 | 

 
 | 
 
 
 Classifier-Based 
 | 
 
 
 Bansal et al. [5] , Cho et al. [21] , Seshadri et al. [92] , Bakr et al. [4] , Zhang et al. [112] , Orgad et al. [75] , Zameshina et al. [111] , Shen et al. [94] , Naik Nushi [71] , Friedrich et al. [33] , Lin et al. [57] , Zhang et al. [114] , Mannering [63] , Lee et al. [53] , Kim et al. [47] , Friedrich et al. [34] , Hao et al. [39] 
 | 

 
 | 
 
 
 VQA-Based 
 | 
 
 
 Esposito et al. [26] , Luccioni et al. [60] , Cho et al. [21] , Vice et al. [102] , Struppek et al. [97] , Chinchure et al. [20] , Sathe et al. [90] , Wan Chang [104] 
 | 

 
 | 
 
 
 Embedding Distance-Based 
 | 
 
 
 Bianchi et al. [9] , Vice et al. [102] , Naik Nushi [71] , Li et al. [54] , Basu et al. [8] , Hao et al. [39] , Jha et al. [43] , Lee et al. [53] , He et al. [40] 
 | 

 
 
 
 Embedding-Based Metrics 
 | 
 
 
 Association with Bias Characteristics 
 | 
 
 
 Wang et al. [105] , Mandal et al. [62] , Struppek et al. [97] , Luccioni et al. [60] 
 | 

 
 | 
 
 
 Image Quality 
 | 
 
 
 Naik Nushi [71] , Lee et al. [53] 
 | 

 

 Table 2: Literatures on bias evaluation metrics for T2I models, stratified by metric types and categories. While many studies employed classification-based metrics, only a few used embedding-based methods. 
 
 
 

## Appendix C Summarization of Mitigation Metrics

 
 Table 3 summarizes different types of mitigation approaches in previous works.
While not many have explored bias mitigation in T2I models, most works in this direction explored Parameter-Efficient Finetuning and prompt-based methods.
Other approaches like model editing, inference-time guidance and sampling, and data augmentation remain under-explored.

 
 
 
 
 
 
 
 Category 
 | 
 
 
 Conceptualization 
 | 
 
 
 Works 
 | 

 
 
 
 
 
 Model Weight Refinement 
 | 
 
 
 Finetuning 
 | 
 
 
 Struppek et al. [97] , Esposito et al. [26] , Liu et al. [59] 
 | 

 
 | 
 
 
 Parameter-Efficient Finetuning 
 | 
 
 
 Li et al. [54] , Kim et al. [47] , Shen et al. [94] , Zhang et al. [112] 
 | 

 
 | 
 
 
 Model Editing 
 | 
 
 
 Orgad et al. [75] 
 | 

 
 
 
 Inference-Time and Data Approaches 
 | 
 
 
 Prompt-Based Mitigation 
 | 
 
 
 Bansal et al. [5] , Fraser et al. [31] , Bianchi et al. [9] , Naik Nushi [71] , Friedrich et al. [34] , Wan Chang [104] 
 | 

 
 | 
 
 
 Guided Generation 
 | 
 
 
 Friedrich et al. [33] , He et al. [40] 
 | 

 
 | 
 
 
 Diverse Sampling 
 | 
 
 
 Zameshina et al. [111] 
 | 

 
 | 
 
 
 Data Augmentation 
 | 
 
 
 Esposito et al. [26] 
 | 

 

 Table 3: Literatures on bias mitigation approaches for T2I models that we reviewed, stratified by mitigation types and methods. Most prior works employ Parameter-Efficiant Finetuning and Prompt-Based Mitigation, whereas methods like data augmentation and model editing remain under-explored.