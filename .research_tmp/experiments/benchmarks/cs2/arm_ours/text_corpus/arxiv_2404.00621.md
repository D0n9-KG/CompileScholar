Multimodal Pretraining, Adaptation, and Generation for Recommendation: A Survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2404.00621v2 [cs.IR] 03 Jul 2024 
 
 

# Multimodal Pretraining, Adaptation, and Generation 
 for Recommendation: A Survey

 Conference:  Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining; August 25–29, 2024; Barcelona, Spain Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’24), August 25–29, 2024, Barcelona, Spain DOI:  10.1145/3637528.3671473 ISBN:  979-8-4007-0490-1/24/08 
 
 
 Qijiong Liu
 
 Note:   Equal contribution. Correspondence to: Jieming Zhu.
 
 Affiliation:  The HK PolyU , Hong Kong , China 
 
 email: liu@qijiong.work 
 
 , 
 Jieming Zhu
 
 Affiliation:  Huawei Noah’s Ark Lab , Shenzhen , China 
 
 email: jiemingzhu@ieee.org 
 
 , 
 Yanting Yang
 
 Note:   The work was done when the authors were visiting at Huawei Noah’s Ark Lab.
 
 Affiliation:  Zhejiang Unversity , Hangzhou , China 
 
 email: yantingyang@zju.edu.cn 
 
 , 
 Quanyu Dai
 
 Affiliation:  Huawei Noah’s Ark Lab , Shenzhen , China 
 
 email: daiquanyu@huawei.com 
 
 , 
 Zhaocheng Du
 
 Affiliation:  Huawei Noah’s Ark Lab , Shenzhen , China 
 
 email: zhaochengdu@huawei.com 
 
 , 
 Xiao-Ming Wu
 
 Affiliation:  The HK PolyU , Hong Kong , China 
 
 email: xiao-ming.wu@polyu.edu.hk 
 
 , 
 Zhou Zhao
 
 Affiliation:  Zhejiang Unversity , Hangzhou , China 
 
 email: zhaozhou@zju.edu.cn 
 
 , 
 Rui Zhang
 
 Affiliation:  Huazhong University of Science and Technology, China 
 
 email: rayteam@yeah.net 
 
 and 
 Zhenhua Dong
 
 Affiliation:  Huawei Noah’s Ark Lab , Shenzhen , China 
 
 email: dongzhenhua@huawei.com 
 
 © acmlicensed 

 Abstract. 
 
 Personalized recommendation serves as a ubiquitous channel for users to discover information tailored to their interests. However, traditional recommendation models primarily rely on unique IDs and categorical features for user-item matching, potentially overlooking the nuanced essence of raw item contents across multiple modalities such as text, image, audio, and video. This underutilization of multimodal data poses a limitation to recommender systems, especially in multimedia services like news, music, and short-video platforms. The recent advancements in large multimodal models offer new opportunities and challenges in developing content-aware recommender systems. This survey seeks to provide a comprehensive exploration of the latest advancements and future trajectories in multimodal pretraining, adaptation, and generation techniques, as well as their applications in enhancing recommender systems. Furthermore, we discuss current open challenges and opportunities for future research in this dynamic domain. We believe that this survey, alongside the curated resources 1 1 
 1 
 
 
 
  Github repository: https://mmrec.github.io/survey , will provide valuable insights to inspire further advancements in this evolving landscape.

 
 
 
 Keywords:  Recommender Systems, Multimodal Pretraining, Multimodal Adaptation, Multimodal Generation
 
 

## 1. Introduction

 
 Recommender systems have been widely employed in various online applications, including e-commerce websites, advertising systems, streaming services, and social media platforms, to deliver personalized recommendations to users. Their primary goal is to enhance user experience, boost user engagement, and facilitate the discovery of items tailored to individual interests. However, traditional recommendation models primarily rely on unique IDs (e.g., user/item IDs) and categorical features (e.g., tags) for user-item matching  ( Zhu et al., 2022 ) , potentially overlooking the nuanced essence of raw item contents across multiple modalities such as text, image, audio, and video  ( Yuan et al., 2023 ) . This underutilization of multimodal data poses a limitation to recommender systems, especially in multimedia services like news, music, and short-video platforms  ( Deldjoo et al., 2020 ) .

 
 
 To tackle this limitation, researchers have extensively investigated multimodal recommendation techniques for over a decade, resulting in a large body of research work that explores the integration of multimodal item features into recommendation models. For a comprehensive review, interested readers can refer to recent surveys  ( Zhou et al., 2023b ; Liu et al., 2023a ; Deldjoo et al., 2020 ; Malitesta et al., 2023 ) . These surveys primarily delve into techniques such as multimodal feature extraction  ( Malitesta et al., 2023 ) , feature representation  ( Deldjoo et al., 2020 ) , feature interaction  ( Liu et al., 2023a ) , feature alignment  ( Deldjoo et al., 2020 ) , feature enhancement  ( Liu et al., 2023a ) , and multimodal fusion  ( Zhou et al., 2023b ) for recommendation models. However, the majority of these approaches rely on extracted multimodal feature embeddings, leaving other aspects of multimodal pretraining and generation relatively unexplored. Nowadays, pretrained large models have gained significant popularity in the domains of natural language processing (NLP), computer vision (CV), and multimodal systems (MM). The emergence of language models like the GPT  ( Brown et al., 2020 ) and Llama  ( Touvron et al., 2023b ) series has ushered in a new era of capabilities for understanding and generating language, while the CV field has witnessed breakthroughs with models such as ViT  ( Dosovitskiy et al., 2021 ) and DINOv2  ( Oquab et al., 2023 ) . Leveraging these successes in unimodal domains, the multimodal community has concentrated on on aligning content across different modalities, such as CLIP  ( Radford et al., 2021 ) , CLAP  ( Elizalde et al., 2023 ) and BLIP-2  ( Li et al., 2023b ) . Notably, the recent introduction of groundbreaking technologies, such as ChatGPT  ( OpenAI, 2023a ) , SD  ( Rombach et al., 2022 ) , and Sora  ( Liu et al., 2024d ) , has further advanced the generation capabilities of pretrained large models to unprecedented levels. These recent advancements in pretrained large multimodal models offer new opportunities and challenges in developing content-aware recommender systems.

 
 
 
 
 
 
 
 (a) Multimodal Pretraining. 
 
 
 (b) Multimodal Adaptation. 
 
 
 (c) Multimodal Generation. 
 
 Figure 1 . An overview of multimodal pretraining, adaptation, and generation tasks for recommendation. 
 
 
 In this survey, our objective is to provide a comprehensive overview of multimodal recommendation techniques from a new perspective, focusing on leveraging pretrained multimodal models. We explore the latest advancements and future trajectories in multimodal pretraining, adaptation, and generation techniques, along with their applications to recommender systems. Different from prior works, our survey capitalizes on recent progress in multimodal language models  ( Li et al., 2023b ; OpenAI, 2023b ) , prompt and adapter tuning  ( Li and Liang, 2021 ; Zhou et al., 2022 ) , and generation techniques such as stable diffusion  ( Rombach et al., 2022 ) . Additionally, we delve into the most recent practical developments and remaining open challenges in applying pretrained multimodal models for recommendation tasks.

 
 
 More specifically, Section 2 introduces the task of multimodal pretraining for recommendation, emphasizing methods to enhance in-domain multimodal pretraining using domain-specific data. Section 3 examines multimodal adaptation for recommendation, elucidating how pretrained multimodal models can be adapted to downstream recommendation tasks through techniques such as representation transfer, model finetuning, adapter tuning, and prompt tuning. Section 4 delves into the emerging topic of multimodal generation for recommendation, with a focus on the application of AI-generated content (AIGC) techniques in recommendation contexts. In Section 5 , we outline a range of common applications that necessitate multimodal recommendation, followed by a discussion of open challenges and opportunities for future research in Section 6 . Finally, we conclude the survey in Section 7 . We hope that this survey, along with the curated resources, will inspire further research efforts to advance this evolving landscape.

 
 
 

## 2. Multimodal Pretraining for Recommendation

 
 In contrast to supervised learning directly on domain-specific data, self-supervised pretraining learns from a large-scale unlabeled corpus and then adapts the pretrained model to downstream tasks. This approach allows for the acquisition of rich external knowledge in pretraining data, thus leading to the widespread recognition of its effectiveness.
In this section, we will first provide a review of major pretraining paradigms and then introduce how they are utilized in the recommendation domain. Figure 1(a) presents an overview of multimodal pretraining techniques.

 
 

### 2.1. Self-supervised Pretraining Paradigms

 
 We broadly categorize self-supervised pretraining paradigms into three types according to their pretraining tasks.

 
 
 Reconstructive Paradigm . This pretraining paradigm aims to teach models to reconstruct raw inputs within the information bottleneck framework. Examples include mask prediction methods for partial reconstruction and autoencoder methods for complete reconstruction.
Mask prediction methods were initially introduced in BERT  ( Devlin et al., 2019 ) , where input tokens are randomly masked, prompting the model to learn to predict them based on the surrounding context. In contrast, autoencoder methods (e.g., AE  ( Bank et al., 2020 ) , VAE  ( Kingma and Welling, 2014 ) ) encode input data into a concise latent space and subsequently learn to fully recover the input from this latent representation. These methods have found extensive use in self-supervised pretraining across various domains such as text  ( Lewis et al., 2020 ) , vision  ( Dosovitskiy et al., 2021 ; Van Den Oord et al., 2017 ; He et al., 2022 ) , audio  ( Zeng et al., 2021 ) , and multimodal data  ( Rombach et al., 2022 ; Ramesh et al., 2022 ) . Following their success, researchers have applied the reconstructive pretraining paradigm to recommendation tasks. For instance, methods like mask item prediction in Bert4Rec  ( Sun et al., 2019 ) , mask token prediction in Recformer  ( Li et al., 2023c ) , autoencoder-based item tokenization  ( Rajput et al., 2024 ; Liu et al., 2024b ) , and masked node feature reconstruction in PMGT  ( Liu et al., 2021b ) have emerged. Despite significant progress, relying solely on this reconstructive paradigm may not capture proximity information from user-item interactions effectively. Consequently, these methods are typically complemented with a contrastive learning paradigm in practice.

 
 
 Contrastive Paradigm . This pretraining focuses on pairwise similarity, distinguishing between similar and dissimilar data samples by maximizing distances between negative pairs and minimizing them for positive pairs within a representation space. It has proven effective in enhancing the quality of representations across different domains. Examples such as SimCSE  ( Gao et al., 2021 ) for text, SimCLR  ( Chen et al., 2020 ) for images, CLMR  ( Spijkervet and Burgoyne, 2021 ) for music, and CLIP  ( Radford et al., 2021 ) for multimodal representations highlight its versatility and applicability. Given its ability to capture pairwise similarities, this paradigm finds extensive use in aligning user-item preferences. Applications like MGCL  ( Liu et al., 2023d ) , MMSSL  ( Wei et al., 2023a ) , MMCPR  ( Liu et al., 2022a ) , MSM4SR  ( Zhang et al., 2023b ) , and MISSRec  ( Wang et al., 2023c ) exemplify the utilization of contrastive learning for enhancing multimodal pretraining in recommender systems.

 
 
 Autoregressive Paradigm . This paradigm has recently achieved remarkable success, particularly with the rise of large language models (LLMs) such as the GPT family  ( Radford et al., 2018 ; Radford et al., 2019 ; Brown et al., 2020 ; OpenAI, 2023b ) . It generates sequence data token by token in an autoregressive manner, where each token is predicted based on previous observations. In other words, this approach operates in a unidirectional, left-to-right generation framework, which is different from the reconstructive paradigm that employs bidirectional context to predict masked tokens. It has also gained rapid adoption in the CV domain  ( Ramesh et al., 2021 ) and multimodal domain  ( Zhan et al., 2024 ) . In the realm of recommender systems, user behavior sequences naturally lend themselves to sequential processing, fostering the development of numerous autoregressive sequential recommendation models such as SASRec  ( Kang and McAuley, 2018 ) . Recent studies, such as P5  ( Geng et al., 2022 ) and VIP5  ( Geng et al., 2023 ) , have explored the integration of LLMs or pretrained multimodal models into recommendation tasks. Concurrently, generative recommendation, which frames recommendation as autoregressive sequence generation, has emerged as a burgeoning area of research  ( Rajput et al., 2024 ; Wang et al., 2024b ; Wei et al., 2024a ) .

 
 
 

### 2.2. Content-aware Pretraining for Recommendation

 
 Content-aware recommender systems strive to incorporate the semantic content of items to improve recommendation accuracy. Consequently, numerous studies have explored content-enhanced pretraining methods for recommendation systems. In this section, we categorize existing research based on the modalities employed for pretraining and discuss their application both generally and within recommendation systems.

 
 
 Text-based Pretraining .
Texts are among the most prevalent forms of content in recommender systems, applied in contexts such as news recommendation and review-based recommendation. Within the domain of natural language processing (NLP), pretrained language models like BERT  ( Devlin et al., 2019 ) and T5  ( Raffel et al., 2020 ) have been developed to capture context-aware representations of text. These models typically follow a pretraining-finetuning paradigm tailored to specific tasks. Recently, large language models (LLMs) such as ChatGPT  ( OpenAI, 2023a ) and LLaMa  ( Touvron et al., 2023a ) have demonstrated significant capabilities in language-related tasks, leveraging techniques such as prompting and in-context learning. Building on their success, text-enhanced pretraining has gained traction in recommender systems. Notable examples include MINER  ( Li et al., 2022 ) for news recommendation, Recformer  ( Li et al., 2023c ) for sequential recommendation, UniSRec  ( Hou et al., 2022 ) for cross-domain recommendation, and P5  ( Geng et al., 2022 ) for LLM-based interactive recommendation.

 
 
 Audio-based Pretraining .
Music recommendation represents a prominent scenario heavily reliant on audio modalities to capture content semantics. Analogous to the NLP domain, various pretraining techniques have been employed to enhance audio representations, including Wav2Vec  ( Schneider et al., 2019 ) , MusicBert  ( Zeng et al., 2021 ) , MART  ( Yao et al., 2024 ) , and MERT  ( Li et al., 2023d ) . In the context of music recommendation, researchers explore leveraging these audio pretraining methods by utilizing user-item interactions as supervision signals to finetune music representations. For example, Chen et al. ( Chen et al., 2021a ) propose learning user-audio embeddings from track data and user interests through contrastive learning techniques. Furthermore, Huang et al. ( Huang et al., 2020 ) integrate pairwise textual and audio features into a convolutional model to jointly learn content embeddings in a similarity metric. Interested readers can find additional examples in a comprehensive review paper  ( Deldjoo et al., 2021 ) .

 
 
 Vision-based Pretraining . Images and videos constitute the primary visual data in multimedia recommendation scenarios as ads, movies, and videos. In the CV domain, the evolution of vision-based pretraining has transitioned from CNN-based architectures like ResNet  ( He et al., 2016 ) to transformer architectures such as ViT  ( Dosovitskiy et al., 2021 ) and DINOv2  ( Oquab et al., 2023 ) , enabling the extraction of versatile visual features. These pretrained models have significantly advanced vision-aware recommendation systems. Researchers like Liu et al. ( Liu et al., 2020a ) and Chen et al. ( Chen et al., 2022 ) leverage pretrained CNN encoders with category priors to extract image features for industrial recommendation tasks, finetuning image encoders alongside recommendation models. Similarly, Wang et al. ( Wang et al., 2023c ) and Wei et al. ( Wei et al., 2024a ) utilize pretrained transformer encoders, such as ViT  ( Radford et al., 2021 ) , to encode images and sequences of user behaviors. Vision foundation models continue to evolve rapidly, promising future applications in recommendation tasks. Looking ahead, there is a growing interest in exploring the use of newly pretrained models for recommendation tasks. For instance, leveraging pretrained video transformers like Video-LLaVA  ( Lin et al., 2023a ) remains an unexplored area in building video recommendation models.

 
 
 Multimodal Pretraining . In current literature, most studies tend to focus on modeling the primary modality of content, such as text for news recommendation  ( Li et al., 2022 ) , audio for music recommendation  ( Chen et al., 2021a ) , and images for e-commerce recommendation  ( Liu et al., 2020a ) . However, multimedia content inherently involves multiple modalities. For instance, news articles often include titles, descriptions, and accompanying images. Similarly, video recommendation involves handling visual frames, audio signals, and subtitles.

 
 
 Unlike single-modal techniques, multimodal models must capture both commonalities and complementary information across multimodal data sources through techniques like cross-modal alignment and fusion. In recent years, multimodal pretraining has seen rapid development, resulting in a plethora of pretrained models, including single-stream models (e.g., VL-BERT  ( Su et al., 2019 ) ), dual-stream models (e.g., CLIP  ( Radford et al., 2021 ) ), and hybrid models (e.g., FLAVA  ( Singh et al., 2022 ) and CoCa  ( Yu et al., 2022b ) ). Recent research has also focused on achieving unified representations of multimodal data, exemplified by models like ImageBind  ( Girdhar et al., 2023 ) , MetaTransformer  ( Zhang et al., 2023a ) , and UnifiedIO-2  ( Lu et al., 2023 ) . Another emerging trend involves integrating multimodal encoders with large language models, resulting in multimodal large language models such as BLIP-2  ( Li et al., 2023b ) , Flamingo  ( Alayrac et al., 2022 ) , and Llava  ( Liu et al., 2023c ) . These advancements offer promising opportunities for building modern multimodal recommender systems.

 
 
 Initial efforts in this direction include models like MISSRec  ( Wang et al., 2023c ) , Rec-GPT4V  ( Liu et al., 2024c ) , MMSSL  ( Wei et al., 2023a ) , MSM4SR  ( Zhang et al., 2023b ) , and AlignRec  ( Liu et al., 2024e ) . However, current research primarily focuses on utilizing off-the-shelf pretrained multimodal encoders and integrating them with sequence-based or graph-based pretraining techniques for handling recommendation data. There remains a gap in how to specifically pretrain domain-specific multimodal models tailored for recommendation tasks. Pioneering efforts have been made in the e-commerce domain, as evidenced by models such as M5Product  ( Dong et al., 2022 ) , K3M  ( Zhu et al., 2021 ) , ECLIP  ( Jin et al., 2023 ) , and CommerceMM  ( Yu et al., 2022a ) . Future research is expected to further explore and advance in this direction.

 
 
 
 

## 3. Multimodal Adaption for Recommendation

 
 While most existing pretrained models are trained on general data corpora, adapting them for recommender systems requires strategic methods to fully utilize their learned knowledge. This section summarizes four major adaptation techniques: representation transfer, model finetuning, adapter tuning, and prompt tuning. Each technique provides a distinct approach to harnessing the benefits of a pretrained model. Figure 1(b) presents an overview of multimodal adaptation techniques.

 
 

### 3.1. Representation Transfer

 
 Representation transfer is one of the most commonly used adaptation techniques for transferring pretrained knowledge to recommendation models. Specifically, item representations are extracted from frozen pretrained models and used as additional features alongside ID embeddings. These representations provide supplementary general information to recommender systems, addressing the cold-start problem where new or infrequently interacted items may have inadequate ID embeddings derived from limited interactions  ( Liu et al., 2022b ) . Approaches based on representation transfer have been extensively studied and proven effective across various domains, e.g., text-based recommendation  ( Wu et al., 2019 ) , vision-based recommendation  ( Shang et al., 2023 ; Baltescu et al., 2022 ) , multimodal recommendation  ( Xun et al., 2021 ; Hu et al., 2024 ; Du et al., 2020 ; Wei et al., 2023b ; Liu et al., 2021a ) . For multimodal scenarios specifically, significant efforts have been directed towards fusing representations from multiple modalities. This includes techniques such as early fusion  ( Liu et al., 2019 ; Hu et al., 2024 ; Liu et al., 2021b ; Zhou and Shen, 2023 ) , intermediate fusion  ( Xun et al., 2021 ; Du et al., 2020 ; Wu et al., 2022 ) , and late fusion  ( Tao et al., 2020 ; Wei et al., 2020 ) . Another research focus involves aligning multimodal representations within user behavior spaces using methods like content-ID alignment  ( Wei et al., 2021 ; Liu et al., 2023e ) , item-to-item matching  ( Huang et al., 2021 ) , user sequence modeling  ( Hou et al., 2022 ) , and graph neural networks  ( Wei et al., 2019 ) .

 
 
 However, straightforward and efficient representation transfer may encounter a significant domain generalization gap due to misalignment between the semantic space and the behavior space, which may not consistently lead to performance improvements in practice. Model finetuning offers a direct solution to address this issue. Furthermore, as noted by KDSR  ( Hu et al., 2024 ) , there is a risk of forgetting modality features in representations, leading them to resemble those trained without incorporating such features. One viable approach involves integrating explicit constraints  ( Hu et al., 2024 ) , while another potential solution introduces semantic tokenization techniques  ( Rajput et al., 2024 ; Singh et al., 2023 ; Liu et al., 2024b ; Jin et al., 2024 ) , which quantize item content representations into discrete tokens.

 
 
 

### 3.2. Model Finetuning

 
 Model finetuning refers to the process of further training a pretrained model on task-specific data. Its goal is to adapt the model parameters to effectively capture domain-specific nuances, thereby improving its performance on the specific downstream task. This pretraining-finetuning paradigm has proven successful in various practical applications. Specifically, finetuning can involve aligning the semantic space of pretrained models with the behavior space of recommendation models. Depending on the application of pretrained models, they can extract item representations  ( Li et al., 2022 ; Xv et al., 2022 ) , user representations  ( Sun et al., 2023 ) , or both  ( Li et al., 2023c ; Liu et al., 2022b ; Wang et al., 2023c ) . Moreover, based on the types of downstream tasks, current research can be classified into representation-based matching tasks  ( Wu et al., 2021 ; Li et al., 2022 ; Wang et al., 2023c ) and scoring-based ranking tasks  ( Zhang et al., 2021 ; Xv et al., 2022 ; Liu et al., 2020a ) .

 
 
 However, end-to-end finetuning of pretrained models with recommendation data faces challenges related to training efficiency. Recommendation tasks often require processing millions or even billions of samples daily. Fully finetuning a pretrained model substantially amplifies training overhead, which poses practical limitations in large-scale recommender systems, particularly given the scale of large language and multimodal models  ( Touvron et al., 2023a ; Li et al., 2023b ) . Moreover, finetuning with large volumes of data easily leads to the issue of catastrophic forgetting, where previously learned knowledge rapidly deteriorates during continual training.

 
 
 

### 3.3. Adapter Tuning

 
 To reduce training overhead with pretrained large models, parameter-efficient finetuning (PEFT) methods have been developed. One prominent approach is through parameter-efficient adapters like LoRAs  ( Hu et al., 2022 ) , which integrate compact, task-specific modules directly into pretrained models. This strategy effectively reduces the number of parameters needed for finetuning and facilitates rapid model adaptation. Widely recognized for its efficacy across various domains, PEFT techniques have gained significant traction in recommendation systems  ( Liu et al., 2024a ; Hou et al., 2022 ; Fu et al., 2024 ; Xi et al., 2023 ; Geng et al., 2023 ) . For example, the ONCE framework  ( Liu et al., 2024a ) leverages the pretrained Llama model  ( Touvron et al., 2023a ) with LoRAs as item encoders to enhance content-aware recommendation. Similarly, UniSRec  ( Hou et al., 2022 ) employs an MOE-based adapter with the BERT model to improve semantic representations of items across diverse domains. In the realm of multimodal recommendation, TransRec  ( Fu et al., 2024 ) and VIP5  ( Geng et al., 2023 ) have introduced layerwise adapters. M3SRec utilizes modality-specific MOE adapters  ( Bian et al., 2023 ) , while EM3 employs multimodal fusion adapters  ( Deng et al., 2024 ) during finetuning. Nonetheless, the ongoing challenge lies in designing adapters that effectively balance both effectiveness and efficiency, which remains an active area of research.

 
 
 

### 3.4. Prompt Tuning

 
 With the emergence of large language models, prompting has become a pivotal technique in harnessing their capabilities to generate desired outputs or perform specific tasks  ( Liu et al., 2023g ) . Instead of using handcrafted prompts, prompt tuning aims to learn task-adaptable prompts from task-specific data while keeping the model parameters frozen. As a result, prompt tuning can avoid catastrophic forgetting and enable fast adaptation with only prompt tokens as tunable parameters. Depending on whether prompts are optimized in a discrete token space, they can be further categorized into hard prompt tuning, such as AutoPrompt  ( Shin et al., 2020 ) , and soft prompt tuning, like Prefix-Tuning  ( Li and Liang, 2021 ) . Prompt tuning has been successful in visual learning  ( Zhou et al., 2022 ) and multimodal learning  ( Duan et al., 2023 ; Khattak et al., 2023 ) , enhancing model performance.

 
 
 For multimodal recommendation tasks, prompt tuning has emerged as a novel technique to adapt pretrained models. Recent studies such as RecPrompt  ( Liu et al., 2023f ) , ProLLM4Rec  ( Xu et al., 2024 ) , Prompt4NR  ( Zhang and Wang, 2023 ) , and PBNR  ( Li et al., 2023f ) employ prompting methods to customize large language models (LLMs) for news recommendation tasks. Additionally, DeepMP  ( Wei et al., 2024a ) , VIP5  ( Geng et al., 2023 ) , and PromptMM  ( Wei et al., 2024b ) employ prompt tuning to integrate and adapt multimodal content knowledge to enhance recommendation. Despite recent advancements, this area of research remains underexplored. An interesting direction is the development of personalized multimodal prompting techniques to advance multimodal recommendation systems.

 
 
 
 

## 4. Multimodal Generation for Recommendation

 
 With recent advancements in generative models, AI-generated content (AIGC) has gained significant popularity across diverse applications. In this section, we explore potential research avenues for employing AIGC techniques within recommender systems. Figure 1(c) presents an overview of multimodal generation techniques.

 
 

### 4.1. Text Generation

 
 With the support of powerful large language models (LLMs), text generation has become a mature capability and is now being applied in various tasks within the recommendation domain  ( Murakami et al., 2023 ) .

 
 
 
 • 
 
 Keyword Generation : Keyword tagging plays a pivotal role in content understanding for ads targeting and recommendation. Previous techniques mostly rely on explicit keyword extraction from textual content, potentially missing important keywords absent from the text. Consequently, keyword generation techniques have been widely applied to enhance the keyword tagging process  ( Song et al., 2023a ; Li et al., 2023a ) .

 

 • 
 
 News Headline Generation : The demand for personalized and engaging news content has fueled the exploration of news headline generation. Conventionally, headline generation is framed as a text summarization task, condensing input text or multimodal content into a title  ( Krubinski and Pecina, 2024 ; Ding et al., 2023 ) . However, typical news headlines may lack appeal or relevance to specific users, prompting the need for personalized approaches. Consequently, personalized headline generation has emerged as a compelling research topic, focusing on generating titles tailored to individual users’ reading preferences and available news content  ( Gu et al., 2020 ; Salemi et al., 2023 ) .

 

 • 
 
 Marketing Copy Generation : Marketing copy refers to the text used to promote a product and motivate consumers to purchase. It plays a vital role in capturing users’ interest and enhancing engagement. Recent efforts have focused on automatic marketing copywriting based on LLMs  ( Mita et al., 2023 ; Zhou et al., 2024 ) 

 

 • 
 
 Explanation Generation : In interactive scenarios, the demand for explainable recommendations is growing significantly. This involves generating natural language explanations to justify the recommendation of items to individual users, thereby enhancing user understanding and trust in the system  ( Li et al., 2023e ; Zhao et al., 2019 ) .

 

 • 
 
 Dialogue Generation : Dialogue generation  ( Yang et al., 2022 ) is essential in conversational recommender systems, encompassing the generation of responses that describe recommended items  ( Feng et al., 2023 ) . Moreover, it entails generating questions to guide users towards further rounds of conversation and interaction  ( Wang et al., 2023a ) .

 

 
 
 
 While these tasks benefit from powerful LLMs, two critical challenges persist in text generation for recommendation: 1) Controllable Generation: Industrial applications necessitate precise control over generated texts to ensure correctness of product descriptions, use unique selling propositions, or adhere to specific writing styles  ( Zhou et al., 2023a ) . 2) Knowledge-Enhanced Generation: Existing LLMs often lack explicit awareness of domain-specific knowledge, such as product entities, categories, and selling points. Recent research has concentrated on integrating domain-specific knowledge bases to achieve more satisfactory results  ( Xu et al., 2021 ; Trisedya et al., 2022 ) .

 
 
 

### 4.2. Image and Video Generation

 
 Text-to-image generation has achieved remarkable success with the prevalence of diffusion models (e.g., SD  ( Rombach et al., 2022 ) ). In this section, we delve into their potential applications in e-commerce and advertising. Unlike natural image generation, generating product images and ad banners involves dealing with complex layouts, encompassing various elements such as products, logos, and textual descriptions. Consequently, unique challenges arise in designing a coherent layout and effectively integrating text with appropriate fonts and colors to create visually appealing posters.

 
 
 Specifically, Inoue et al.  ( Inoue et al., 2023 ) propose LayoutDM, a model designed to effectively handle structured layout data and facilitate the discrete diffusion process. Hsu et al.  ( Hsu et al., 2023 ) enable content-aware layout generation (namely PosterLayout) by arranging predefined spatial elements on a given canvas. Lin et al. ( Lin et al., 2023b ) develop AutoPoster, a highly automated and content-aware system for generating advertising posters. Concurrently, some studies explore text design for poster generation. For example, Gao et al.  ( Gao et al., 2023 ) introduce TextPainter, a novel multimodal approach that leverages contextual visual information and corresponding text semantics to generate text images. Tuo et al.  ( Tuo et al., 2023 ) propose a diffusion-based multilingual visual text generation and editing model, AnyText, which addresses how to render accurate and coherent text in the image.

 
 
 More recently, video generation has made significant strides. Sora  ( Liu et al., 2024d ) emerges as a groundbreaking technology showcasing immense potential for generating advertising videos for products. In this context, Gong et al. ( Gong et al., 2024 ) introduce AtomoVideo, a high-fidelity image-to-video generation solution that effectively transforms product images into engaging promotional videos for advertising purposes. Additionally, Liu et al. ( Liu et al., 2020b ) have devised a system capable of automatically generating visual storylines from a given set of visual materials, producing compelling promotional videos tailored for e-commerce. Furthermore, Wang et al.  ( Wang et al., 2024a ) have developed an integrated approach, merging text-to-image models, video motion generators, reference image embedding modules, and frame interpolation modules into an end-to-end video generation pipeline, which is valuable for micro-video recommendation platforms. We believe that this field is rapidly expanding, enabling the advancement of AIGC-based recommendation and advertising applications.

 
 
 

### 4.3. Personalized Generation

 
 With the rise of AIGC, there is a notable shift towards personalized generation, aiming to enhance the customization and personalization of generated content. This trend holds particular significance in recommendation scenarios, where personalized content can better cater to users’ interests. Pioneering work has been undertaken in various domains, including personalized news headline generation  ( Ao et al., 2021 ; Ao et al., 2023 ; Salemi et al., 2023 ; Cai et al., 2023 ) , personalized product description generation in e-commerce  ( Deng et al., 2022 ) , personalized answer generation  ( Deng et al., 2022 ) , personalized image generation with identity preservation  ( Dosovitskiy et al., 2021 ) , and personalized multimodal generation  ( Shen et al., 2024 ) . Integrating recommender systems with personalized generation techniques shows promise for developing next-generation recommender systems.

 
 
 
 

## 5. Applications

 
 In this section, we summarize some common application domains that require multimodal recommendation techniques.

 
 
 
 • 
 
 E-commerce Recommendation . E-commerce represents one of the most extensively studied application domains in recommender systems research, aimed at assisting users in discovering items they are likely to purchase. The abundance of multimodal data in e-commerce, including product titles, descriptions, images, and reviews, poses a challenge in integrating different modalities with user interaction data to enhance recommendation quality. To address this challenge, numerous research efforts have been undertaken. Notable examples include works by Alibaba  ( Ge et al., 2018 ; Xv et al., 2022 ; Li et al., 2020 ) , JD.com  ( Xiao et al., 2022a ; Liu et al., 2020a ) , and Pinterest  ( Baltescu et al., 2022 ) .

 

 • 
 
 Advertisement Recommendation . Online advertising serves as a primary revenue source for many web applications. Advertising creatives play a pivotal role in this ecosystem, spanning various formats such as images, titles, and videos. Aesthetic creatives have the potential to engage potential users and enhance the click-through rate (CTR) of products  ( Chen et al., 2021b ) . There is also a pressing need to understand ad creatives better to effectively align advertisements with users’ interests  ( Yang et al., 2019 ; Yang et al., 2023 ) .

 

 • 
 
 News Recommendation . Personalized news recommendation is a crucial technique for assisting users in discovering news of interest. To enhance recommendation accuracy and diversity, recommender systems must comprehend news content and extract semantic information from a user’s reading history. This often involves learning semantic representations of news titles, abstracts, body text, and cover images. Recent research has focused on modeling features from multiple modalities, as exemplified by MM-Rec  ( Wu et al., 2022 ) and IMRec  ( Xun et al., 2021 ) .

 

 • 
 
 Video Recommendation .
With the surge in popularity of micro-video platforms, video recommendation has garnered significant attention within the community. Videos encapsulate a multitude of modalities, including titles, thumbnail images, frames, audio tracks, transcripts, and more. Current research efforts have been concentrated on integrating and adapting multimodal information within micro-video recommendation models.  ( Wei et al., 2019 ; Yi et al., 2022 ) . Notably, Ni et al.  ( Ni et al., 2023 ) have recently introduced a comprehensive micro-video recommendation dataset, enriched with abundant multimodal side information, to foster further research in this domain.

 

 • 
 
 Music Recommendation .
The realm of music streaming services represents another prominent domain that necessitates multimodal recommendation techniques. Within this sphere, a diverse array of multimodal data is involved, including music audio, scores, lyrics, tags, and reviews. Leveraging these various types of music data has proven effective in crafting more personalized recommendations aimed at engaging users; notable examples can be found in  ( Huang et al., 2020 ; Chen et al., 2021a ) . Additionally, Shen et al.  ( Shen et al., 2020 ) propose that incorporating multimodal information from users’ social media can offer insights into their personalities, emotions, and mental well-being, thereby enhancing the accuracy of music recommendation.

 

 • 
 
 Fashion Recommendation . With the visual and aesthetic nature of fashion products, fashion recommendation has emerged as a distinct vertical domain. Unlike traditional recommender systems, fashion recommendation not only suggests individual items but also outfits that complement multiple items. Multimodal understanding capabilities play a pivotal role in this area, including tasks such as localizing fashion items from images, identifying their attributes, and computing compatibility scores for multiple items  ( Deldjoo et al., 2024 ; Song et al., 2023b ) . Moreover, pioneering work  ( Zhu et al., 2023 ) has developed text-to-image diffusion models that allow users to virtually try on clothes. These techniques are expected to enhance the personalization of fashion recommendation and elevate user experience to the next level.

 

 • 
 
 LBS Recommendation . Location-based services (LBS) have become ubiquitous, offering a wide range of services including taxi travel, food delivery, and restaurant recommendation. In these contexts, users can share their Points of Interest (POI) check-ins, photos, opinions, and comments, which encompass a rich array of multimodal spatio-temporal data. Integrating this multimodal information and understanding spatio-temporal correlations among locations enables more accurate modeling of user preferences. Notable examples can be found in  ( Qin et al., 2022 ; Liu et al., 2023b ) .

 

 
 
 
 

## 6. Challenges and Opportunities

 
 In this section, we discuss the
persistent challenges and emerging opportunities for future research.

 
 • 
 
 Multimodal Information Fusion . Multimodal fusion has been extensively explored in research. Within recommender systems, current studies primarily concentrate on fusing and adapting multimodal feature embeddings of items to recommendation models  ( Zhou et al., 2023b ) . However, multimodal information for recommendation inherently adopts a hierarchical structure, ranging from user behavior sequences to individual items, each comprising multiple modalities and further subdivided into semantic tokens and objects. Additionally, the impact of information from diverse modalities and regions can vary significantly among different users. As a result, the challenge lies in effectively fusing multimodal information in a hierarchical and personalized manner to optimize recommendations.

 

 • 
 
 Multimodal Multi-domain Recommendation. 
Multimodal information provides rich semantic insights into item content. Despite considerable research into multimodal recommendation and cross-domain recommendation, effectively leveraging multimodal information to bridge the information gap across domains remains an open challenge  ( Sun et al., 2023 ) . For instance, recommending music based on a user’s reading habits entails semantic alignment across modalities (audio vs. text) and domains (music vs. books).

 

 • 
 
 Multimodal Foundation Models for Recommendation. 
While large language models and large multimodal models have emerged as foundation models in the NLP and CV domains, there exists a compelling opportunity to extend this exploration into the recommendation domain. An ideal recommendation foundation model should demonstrate robust in-context learning capabilities while maintaining generalizability across diverse tasks and domains  ( Huang et al., 2024 ) . Potential avenues for exploration include adapting existing multimodal LLMs for recommendation tasks (e.g., ( Geng et al., 2023 ) ), or conducting the pretraining of a multimodal generative model from scratch using large-scale multimodal multi-domain recommendation data.

 

 • 
 
 AIGC for Recommendation. 
The integration of AIGC represents a notable advancement in recommender systems, offering an opportunity to significantly enhance user personalization, engagement, and overall experience. This encompasses personalized news headlines, tailored advertising creatives, and explanatory content across diverse recommendation contexts. This field is rapidly expanding, with the primary challenge lying in achieving a comprehensive understanding of both content and users, facilitating controllable generation, and ensuring accurate formatting to optimize the user experience. Additionally, it is imperative to address potential ethical and privacy concerns arising from the use of AIGC.

 

 • 
 
 Multimodal Recommendation Agent. 
LLM-based agents  ( Wang et al., 2023b ) have demonstrated exceptional proficiency in automating tasks through extensive knowledge and strong reasoning capabilities. The integration of these agents has introduced innovative prospects in the field of recommendation, particularly in conversational recommendation  ( Fang et al., 2024 ) . This entails directly engaging users in the task completion process, thereby enhancing the user experience and the effectiveness of recommender systems. As a concrete example, integrating conversation and virtual try-on generation  ( Zhu et al., 2023 ) capabilities may present new opportunities for fashion recommendation.

 

 • 
 
 Efficiency of Training and Inference. 
Recommendation tasks typically have stringent latency requirements to meet real-time service demands. Therefore, ensuring training and inference efficiency becomes imperative when applying multimodal pretraining and generation techniques in practice. There is a high demand for the development of efficient strategies to leverage the capabilities of multimodal models. Pioneer efforts in this direction include speeding up training by merging item sets to avoid redundant encoding operations  ( Xiao et al., 2022b ) and enhancing inference speed  ( Liu et al., 2022b ; Chen et al., 2022 ) through caching item and user representations.

 

 
 
 
 

## 7. Conclusion

 
 Multimodal recommendation, an immensely promising field, has garnered significant attention in recent years, fueled by advancements in both multimodal machine learning and the recommendation system community. The advent of large multimodal models has transformed the multimodal recommendation landscape, endowing it with enhanced capabilities for comprehension and content generation. This paper provides a systematical overview of the current multimodal recommendation framework, focusing on key aspects such as multimodal pretraining, adaptation, and generation. Additionally, we delve into its applications, challenges, and future prospects. Our aim is to offer this survey as a resourceful guide to aid subsequent research in the field.

 
 
 Acknowledgements. 
We thank Dr. Xin Zhou and Dr. Chuhan Wu for the discussion and contribution to the tutorial materials of multimodal pretraining and generation for recommendation presented at WWW 2024  ( Zhu et al., 2024 ) .

 
 
 

## References

 
 
 Alayrac et al . (2022) 
 
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al . 2022.

 
 Flamingo: a visual language model for few-shot learning.

 
 Advances in Neural Information Processing Systems (NeurIPS) (2022), 23716–23736.

 
 
 

 
 Ao et al . (2023) 
 
Xiang Ao, Ling Luo, Xiting Wang, Zhao Yang, Jiun-Hung Chen, Ying Qiao, Qing He, and Xing Xie. 2023.

 
 Put Your Voice on Stage: Personalized Headline Generation for News Articles.

 
 TKDD 18, 3 (2023).

 
 
 

 
 Ao et al . (2021) 
 
Xiang Ao, Xiting Wang, Ling Luo, Ying Qiao, Qing He, and Xing Xie. 2021.

 
 PENS: A Dataset and Generic Framework for Personalized News Headline Generation. In Proceedings of ACL/IJCNLP . 82–92.

 
 
 

 
 Baltescu et al . (2022) 
 
Paul Baltescu, Haoyu Chen, Nikil Pancha, Andrew Zhai, Jure Leskovec, and Charles Rosenberg. 2022.

 
 Itemsage: Learning product embeddings for shopping recommendations at pinterest. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) . 2703–2711.

 
 
 

 
 Bank et al . (2020) 
 
Dor Bank, Noam Koenigstein, and Raja Giryes. 2020.

 
 Autoencoders.

 
 CoRR abs/2003.05991 (2020).

 
 
 

 
 Bian et al . (2023) 
 
Shuqing Bian, Xingyu Pan, Wayne Xin Zhao, Jinpeng Wang, Chuyuan Wang, and Ji-Rong Wen. 2023.

 
 Multi-modal Mixture of Experts Represetation Learning for Sequential Recommendation. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM) . 110–119.

 
 
 

 
 Brown et al . (2020) 
 
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al . 2020.

 
 Language models are few-shot learners.

 
 Advances in Neural Information Processing Systems (NeurIPS) (2020), 1877–1901.

 
 
 

 
 Cai et al . (2023) 
 
Pengshan Cai, Kaiqiang Song, Sangwoo Cho, Hongwei Wang, Xiaoyang Wang, Hong Yu, Fei Liu, and Dong Yu. 2023.

 
 Generating User-Engaging News Headlines. In Proceedings of ACL . 3265–3280.

 
 
 

 
 Chen et al . (2021b) 
 
Jin Chen, Ju Xu, Gangwei Jiang, Tiezheng Ge, Zhiqiang Zhang, Defu Lian, and Kai Zheng. 2021b.

 
 Automated Creative Optimization for E-Commerce Advertising. In The ACM Web Conference (WWW) . 2304–2313.

 
 
 

 
 Chen et al . (2021a) 
 
Ke Chen, Beici Liang, Xiaoshuan Ma, and Minwei Gu. 2021a.

 
 Learning audio embeddings with user listening data for content-based music recommendation. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) . IEEE, 3015–3019.

 
 
 

 
 Chen et al . (2020) 
 
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. 2020.

 
 A Simple Framework for Contrastive Learning of Visual Representations. In Proceedings of the 37th International Conference on Machine Learning (ICML) . 1597–1607.

 
 
 

 
 Chen et al . (2022) 
 
Xin Chen, Qingtao Tang, Ke Hu, Yue Xu, Shihang Qiu, Jia Cheng, and Jun Lei. 2022.

 
 Hybrid CNN Based Attention with Category Prior for User Image Behavior Modeling. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) . 2336–2340.

 
 
 

 
 Deldjoo et al . (2024) 
 
Yashar Deldjoo, Fatemeh Nazary, Arnau Ramisa, Julian J. McAuley, Giovanni Pellegrini, Alejandro Bellogín, and Tommaso Di Noia. 2024.

 
 A Review of Modern Fashion Recommender Systems.

 
 ACM Comput. Surv. 56, 4 (2024), 87:1–87:37.

 
 
 

 
 Deldjoo et al . (2020) 
 
Yashar Deldjoo, Markus Schedl, Paolo Cremonesi, and Gabriella Pasi. 2020.

 
 Recommender systems leveraging multimedia content.

 
 Comput. Surveys 53, 5 (2020), 1–38.

 
 
 

 
 Deldjoo et al . (2021) 
 
Yashar Deldjoo, Markus Schedl, and Peter Knees. 2021.

 
 Content-driven Music Recommendation: Evolution, State of the Art, and Challenges.

 
 CoRR abs/2107.11803 (2021).

 
 
 

 
 Deng et al . (2024) 
 
Xiuqi Deng, Lu Xu, Xiyao Li, Jinkai Yu, Erpeng Xue, Zhongyuan Wang, Di Zhang, Zhaojie Liu, Guorui Zhou, Yang Song, Na Mou, Shen Jiang, and Han Li. 2024.

 
 End-to-end training of Multimodal Model and ranking Model.

 
 CoRR abs/2404.06078 (2024).

 
 
 

 
 Deng et al . (2022) 
 
Yang Deng, Yaliang Li, Wenxuan Zhang, Bolin Ding, and Wai Lam. 2022.

 
 Toward Personalized Answer Generation in E-Commerce via Multi-perspective Preference Modeling.

 
 ACM Trans. Inf. Syst. 40, 4 (2022), 87:1–87:28.

 
 
 

 
 Devlin et al . (2019) 
 
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.

 
 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT) . 4171–4186.

 
 
 

 
 Ding et al . (2023) 
 
Zijian Ding, Alison Smith-Renner, Wenjuan Zhang, Joel R. Tetreault, and Alejandro Jaimes. 2023.

 
 Harnessing the power of LLMs: Evaluating human-AI text co-creation through the lens of news headline generation. In Findings of EMNLP . 3321–3339.

 
 
 

 
 Dong et al . (2022) 
 
Xiao Dong, Xunlin Zhan, Yangxin Wu, Yunchao Wei, Michael C. Kampffmeyer, Xiaoyong Wei, Minlong Lu, Yaowei Wang, and Xiaodan Liang. 2022.

 
 M5Product: Self-harmonized Contrastive Learning for E-commercial Multi-modal Pretraining. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) . 21220–21230.

 
 
 

 
 Dosovitskiy et al . (2021) 
 
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021.

 
 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In 9th International Conference on Learning Representations (ICLR) .

 
 
 

 
 Du et al . (2020) 
 
Xiaoyu Du, Xiang Wang, Xiangnan He, Zechao Li, Jinhui Tang, and Tat-Seng Chua. 2020.

 
 How to learn item representation for cold-start multimedia recommendation?. In Proceedings of the 28th ACM International Conference on Multimedia . 3469–3477.

 
 
 

 
 Duan et al . (2023) 
 
Haoyi Duan, Yan Xia, Mingze Zhou, Li Tang, Jieming Zhu, and Zhou Zhao. 2023.

 
 Cross-modal Prompts: Adapting Large Pre-trained Models for Audio-Visual Downstream Tasks. In Advances in Neural Information Processing Systems (NeurIPS) .

 
 
 

 
 Elizalde et al . (2023) 
 
Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang. 2023.

 
 Clap learning audio concepts from natural language supervision. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) . IEEE, 1–5.

 
 
 

 
 Fang et al . (2024) 
 
Jiabao Fang, Shen Gao, Pengjie Ren, Xiuying Chen, Suzan Verberne, and Zhaochun Ren. 2024.

 
 A Multi-Agent Conversational Recommender System.

 
 CoRR abs/2402.01135 (2024).

 
 
 

 
 Feng et al . (2023) 
 
Yue Feng, Shuchang Liu, Zhenghai Xue, Qingpeng Cai, Lantao Hu, Peng Jiang, Kun Gai, and Fei Sun. 2023.

 
 A Large Language Model Enhanced Conversational Recommender System.

 
 CoRR abs/2308.06212 (2023).

 
 
 

 
 Fu et al . (2024) 
 
Junchen Fu, Fajie Yuan, Yu Song, Zheng Yuan, Mingyue Cheng, Shenghui Cheng, Jiaqi Zhang, Jie Wang, and Yunzhu Pan. 2024.

 
 Exploring adapter-based transfer learning for recommender systems: Empirical studies and practical insights. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining (WSDM) . 208–217.

 
 
 

 
 Gao et al . (2021) 
 
Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021.

 
 SimCSE: Simple Contrastive Learning of Sentence Embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP) . 6894–6910.

 
 
 

 
 Gao et al . (2023) 
 
Yifan Gao, Jinpeng Lin, Min Zhou, Chuanbin Liu, Hongtao Xie, Tiezheng Ge, and Yuning Jiang. 2023.

 
 TextPainter: Multimodal Text Image Generation with Visual-harmony and Text-comprehension for Poster Design. In ACM MM . 7236–7246.

 
 
 

 
 Ge et al . (2018) 
 
Tiezheng Ge, Liqin Zhao, Guorui Zhou, Keyu Chen, Shuying Liu, Huiming Yi, Zelin Hu, Bochao Liu, Peng Sun, Haoyu Liu, Pengtao Yi, Sui Huang, Zhiqiang Zhang, Xiaoqiang Zhu, Yu Zhang, and Kun Gai. 2018.

 
 Image Matters: Visually Modeling User Behaviors Using Advanced Model Server. In CIKM . 2087–2095.

 
 
 

 
 Geng et al . (2022) 
 
Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022.

 
 Recommendation as language processing (rlp): A unified pretrain, personalized prompt predict paradigm (p5). In Proceedings of the 16th ACM Conference on Recommender Systems (RecSys) . 299–315.

 
 
 

 
 Geng et al . (2023) 
 
Shijie Geng, Juntao Tan, Shuchang Liu, Zuohui Fu, and Yongfeng Zhang. 2023.

 
 VIP5: Towards Multimodal Foundation Models for Recommendation. In Findings of the Association for Computational Linguistics: EMNLP 2023 . 9606–9620.

 
 
 

 
 Girdhar et al . (2023) 
 
Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. 2023.

 
 Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) . 15180–15190.

 
 
 

 
 Gong et al . (2024) 
 
Litong Gong, Yiran Zhu, Weijie Li, Xiaoyang Kang, Biao Wang, Tiezheng Ge, and Bo Zheng. 2024.

 
 AtomoVideo: High Fidelity Image-to-Video Generation.

 
 (2024).

 
 arXiv:2403.01800

 

 
 Gu et al . (2020) 
 
Xiaotao Gu, Yuning Mao, Jiawei Han, Jialu Liu, You Wu, Cong Yu, Daniel Finnie, Hongkun Yu, Jiaqi Zhai, and Nicholas Zukoski. 2020.

 
 Generating Representative Headlines for News Stories. In The Web Conference 2020 (WWW) . 1773–1784.

 
 
 

 
 He et al . (2022) 
 
Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. 2022.

 
 Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) . 16000–16009.

 
 
 

 
 He et al . (2016) 
 
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016.

 
 Deep residual learning for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) . 770–778.

 
 
 

 
 Hou et al . (2022) 
 
Yupeng Hou, Shanlei Mu, Wayne Xin Zhao, Yaliang Li, Bolin Ding, and Ji-Rong Wen. 2022.

 
 Towards universal sequence representation learning for recommender systems. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) . 585–593.

 
 
 

 
 Hsu et al . (2023) 
 
HsiaoYuan Hsu, Xiangteng He, Yuxin Peng, Hao Kong, and Qing Zhang. 2023.

 
 PosterLayout: A New Benchmark and Approach for Content-Aware Visual-Textual Presentation Layout. In CVPR . 6018–6026.

 
 
 

 
 Hu et al . (2022) 
 
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022.

 
 LoRA: Low-Rank Adaptation of Large Language Models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022 .

 
 
 

 
 Hu et al . (2024) 
 
Hengchang Hu, Qijiong Liu, Chuang Li, and Min-Yen Kan. 2024.

 
 Lightweight Modality Adaptation to Sequential Recommendation via Correlation Supervision.

 
 arXiv preprint arXiv:2401.07257 (2024).

 
 
 

 
 Huang et al . (2024) 
 
Chengkai Huang, Tong Yu, Kaige Xie, Shuai Zhang, Lina Yao, and Julian J. McAuley. 2024.

 
 Foundation Models for Recommender Systems: A Survey and New Perspectives.

 
 CoRR abs/2402.11143 (2024).

 
 
 

 
 Huang et al . (2020) 
 
Qingqing Huang, Aren Jansen, Li Zhang, Daniel PW Ellis, Rif A Saurous, and John Anderson. 2020.

 
 Large-scale weakly-supervised content embeddings for music recommendation and tagging. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) . 8364–8368.

 
 
 

 
 Huang et al . (2021) 
 
Yanhua Huang, Weikun Wang, Lei Zhang, and Ruiwen Xu. 2021.

 
 Sliding spectrum decomposition for diversified recommendation. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery Data Mining (KDD) . 3041–3049.

 
 
 

 
 Inoue et al . (2023) 
 
Naoto Inoue, Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, and Kota Yamaguchi. 2023.

 
 LayoutDM: Discrete Diffusion Model for Controllable Layout Generation. In CVPR . 10167–10176.

 
 
 

 
 Jin et al . (2024) 
 
Mengqun Jin, Zexuan Qiu, Jieming Zhu, Zhenhua Dong, and Xiu Li. 2024.

 
 Contrastive Quantization based Semantic Code for Generative Recommendation.

 
 CoRR abs/2404.14774 (2024).

 
 
 

 
 Jin et al . (2023) 
 
Yang Jin, Yongzhi Li, Zehuan Yuan, and Yadong Mu. 2023.

 
 Learning Instance-Level Representation for Large-Scale Multi-Modal Pretraining in E-Commerce. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) . 11060–11069.

 
 
 

 
 Kang and McAuley (2018) 
 
Wang-Cheng Kang and Julian McAuley. 2018.

 
 Self-attentive sequential recommendation. In IEEE International Conference on Data Mining (ICDM) . IEEE, 197–206.

 
 
 

 
 Khattak et al . (2023) 
 
Muhammad Uzair Khattak, Hanoona Abdul Rasheed, Muhammad Maaz, Salman H. Khan, and Fahad Shahbaz Khan. 2023.

 
 MaPLe: Multi-modal Prompt Learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) . 19113–19122.

 
 
 

 
 Kingma and Welling (2014) 
 
Diederik P. Kingma and Max Welling. 2014.

 
 Auto-Encoding Variational Bayes. In 2nd International Conference on Learning Representations (ICLR) , Yoshua Bengio and Yann LeCun (Eds.).

 
 
 

 
 Krubinski and Pecina (2024) 
 
Mateusz Krubinski and Pavel Pecina. 2024.

 
 Towards Unified Uni- and Multi-modal News Headline Generation. In Proceedings of EACL . 437–450.

 
 
 

 
 Lewis et al . (2020) 
 
Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020.

 
 BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL) . 7871–7880.

 
 
 

 
 Li et al . (2023a) 
 
Chen Li, Yixiao Ge, Jiayong Mao, Dian Li, and Ying Shan. 2023a.

 
 TagGPT: Large Language Models are Zero-shot Multimodal Taggers.

 
 CoRR abs/2304.03022 (2023).

 
 
 

 
 Li et al . (2023b) 
 
Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023b.

 
 Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning (ICML) . 19730–19742.

 
 
 

 
 Li et al . (2023c) 
 
Jiacheng Li, Ming Wang, Jin Li, Jinmiao Fu, Xin Shen, Jingbo Shang, and Julian McAuley. 2023c.

 
 Text is all you need: Learning language representations for sequential recommendation. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) . 1258–1267.

 
 
 

 
 Li et al . (2022) 
 
Jian Li, Jieming Zhu, Qiwei Bi, Guohao Cai, Lifeng Shang, Zhenhua Dong, Xin Jiang, and Qun Liu. 2022.

 
 MINER: Multi-interest matching network for news recommendation. In Findings of the Association for Computational Linguistics (ACL) . 343–352.

 
 
 

 
 Li et al . (2023e) 
 
Lei Li, Yongfeng Zhang, and Li Chen. 2023e.

 
 Personalized Prompt Learning for Explainable Recommendation.

 
 
 
 arXiv:2202.07371

 

 
 Li et al . (2020) 
 
Xiang Li, Chao Wang, Jiwei Tan, Xiaoyi Zeng, Dan Ou, and Bo Zheng. 2020.

 
 Adversarial Multimodal Representation Learning for Click-Through Rate Prediction. In WWW . 827–836.

 
 
 

 
 Li et al . (2023f) 
 
Xinyi Li, Yongfeng Zhang, and Edward C. Malthouse. 2023f.

 
 PBNR: Prompt-based News Recommender System.

 
 CoRR abs/2304.07862 (2023).

 
 
 

 
 Li and Liang (2021) 
 
Xiang Lisa Li and Percy Liang. 2021.

 
 Prefix-Tuning: Optimizing Continuous Prompts for Generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL/IJCNLP) . 4582–4597.

 
 
 

 
 Li et al . (2023d) 
 
Yizhi Li, Ruibin Yuan, Ge Zhang, Yinghao Ma, Xingran Chen, et al . 2023d.

 
 MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training.

 
 CoRR abs/2306.00107 (2023).

 
 
 

 
 Lin et al . (2023a) 
 
Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. 2023a.

 
 Video-LLaVA: Learning United Visual Representation by Alignment Before Projection.

 
 CoRR abs/2311.10122 (2023).

 
 
 

 
 Lin et al . (2023b) 
 
Jinpeng Lin, Min Zhou, Ye Ma, Yifan Gao, Chenxi Fei, Yangjian Chen, Zhang Yu, and Tiezheng Ge. 2023b.

 
 AutoPoster: A Highly Automatic and Content-aware Design System for Advertising Poster Generation. In ACM MM . 1250–1260.

 
 
 

 
 Liu et al . (2021a) 
 
Chang Liu, Xiaoguang Li, Guohao Cai, Zhenhua Dong, Hong Zhu, and Lifeng Shang. 2021a.

 
 Noninvasive self-attention for side information fusion in sequential recommendation. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI) . 4249–4256.

 
 
 

 
 Liu et al . (2020b) 
 
Chang Liu, Han Yu, Yi Dong, Zhiqi Shen, Yingxue Yu, Ian Dixon, Zhanning Gao, Pan Wang, Peiran Ren, Xuansong Xie, Lizhen Cui, and Chunyan Miao. 2020b.

 
 Generating Engaging Promotional Videos for E-commerce Platforms (Student Abstract). In AAAI . 13865–13866.

 
 
 

 
 Liu et al . (2023f) 
 
Dairui Liu, Boming Yang, Honghui Du, Derek Greene, Aonghus Lawlor, Ruihai Dong, and Irene Li. 2023f.

 
 RecPrompt: A Prompt Tuning Framework for News Recommendation Using Large Language Models.

 
 CoRR abs/2312.10463 (2023).

 
 
 

 
 Liu et al . (2023c) 
 
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023c.

 
 Visual Instruction Tuning. In Advances in Neural Information Processing Systems (NeurIPS) .

 
 
 

 
 Liu et al . (2020a) 
 
Hu Liu, Jing Lu, Hao Yang, Xiwei Zhao, Sulong Xu, Hao Peng, Zehua Zhang, Wenjie Niu, Xiaokun Zhu, Yongjun Bao, et al . 2020a.

 
 Category-Specific CNN for Visual-aware CTR Prediction at JD. com. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery Data Mining (KDD) . 2686–2696.

 
 
 

 
 Liu et al . (2023d) 
 
Kang Liu, Feng Xue, Dan Guo, Peijie Sun, Shengsheng Qian, and Richang Hong. 2023d.

 
 Multimodal graph contrastive learning for multimedia-based recommendation.

 
 IEEE Transactions on Multimedia (2023).

 
 
 

 
 Liu et al . (2023g) 
 
Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023g.

 
 Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing.

 
 ACM Comput. Surv. 55, 9 (2023), 195:1–195:35.

 
 
 

 
 Liu et al . (2024a) 
 
Qijiong Liu, Nuo Chen, Tetsuya Sakai, and Xiao-Ming Wu. 2024a.

 
 Once: Boosting content-based recommendation with both open-and closed-source large language models. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining (WSDM) . 452–461.

 
 
 

 
 Liu et al . (2024b) 
 
Qijiong Liu, Hengchang Hu, Jiahao Wu, Jieming Zhu, Min-Yen Kan, and Xiao-Ming Wu. 2024b.

 
 Discrete Semantic Tokenization for Deep CTR Prediction. In Proceedings of the ACM Web Conference (WWW) .

 
 
 

 
 Liu et al . (2023a) 
 
Qidong Liu, Jiaxi Hu, Yutian Xiao, Jingtong Gao, and Xiangyu Zhao. 2023a.

 
 Multimodal Recommender Systems: A Survey.

 
 CoRR abs/2302.03883 (2023).

 
 
 https://doi.org/10.48550/ARXIV.2302.03883 
arXiv:2302.03883

 

 
 Liu et al . (2022b) 
 
Qijiong Liu, Jieming Zhu, Quanyu Dai, and Xiao-Ming Wu. 2022b.

 
 Boosting deep CTR prediction with a plug-and-play pre-trainer for news recommendation. In Proceedings of the 29th International Conference on Computational Linguistics . 2823–2833.

 
 
 

 
 Liu et al . (2019) 
 
Shang Liu, Zhenzhong Chen, Hongyi Liu, and Xinghai Hu. 2019.

 
 User-video co-attention network for personalized micro-video recommendation. In The ACM Web Conference (WWW) . 3020–3026.

 
 
 

 
 Liu et al . (2023b) 
 
Xiaoqian Liu, Xiuyun Li, Yuan Cao, Fan Zhang, Xiongnan Jin, and Jinpeng Chen. 2023b.

 
 Mandari: Multi-Modal Temporal Knowledge Graph-aware Sub-graph Embedding for Next-POI Recommendation.

 
 IEEE International Conference on Multimedia and Expo (ICME) (2023), 1529–1534.

 
 
 

 
 Liu et al . (2024c) 
 
Yuqing Liu, Yu Wang, Lichao Sun, and Philip S. Yu. 2024c.

 
 Rec-GPT4V: Multimodal Recommendation with Large Vision-Language Models.

 
 CoRR abs/2402.08670 (2024).

 
 
 

 
 Liu et al . (2023e) 
 
Yuting Liu, Enneng Yang, Yizhou Dang, Guibing Guo, Qiang Liu, Yuliang Liang, Linying Jiang, and Xingwei Wang. 2023e.

 
 ID Embedding as Subtle Features of Content and Structure for Multimodal Recommendation.

 
 CoRR abs/2311.05956 (2023).

 
 
 

 
 Liu et al . (2021b) 
 
Yong Liu, Susen Yang, Chenyi Lei, Guoxin Wang, Haihong Tang, Juyong Zhang, Aixin Sun, and Chunyan Miao. 2021b.

 
 Pre-training graph transformer with multimodal side information for recommendation. In Proceedings of the 29th ACM International Conference on Multimedia (MM) . 2853–2861.

 
 
 

 
 Liu et al . (2024d) 
 
Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, Lifang He, and Lichao Sun. 2024d.

 
 Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models.

 
 
 
 arXiv:2402.17177

 

 
 Liu et al . (2024e) 
 
Yifan Liu, Kangning Zhang, Xiangyuan Ren, Yanhua Huang, Jiarui Jin, Yingjie Qin, Ruilong Su, Ruiwen Xu, and Weinan Zhang. 2024e.

 
 An Aligning and Training Framework for Multimodal Recommendations.

 
 CoRR abs/2403.12384 (2024).

 
 
 

 
 Liu et al . (2022a) 
 
Zhuang Liu, Yunpu Ma, Matthias Schubert, Yuanxin Ouyang, and Zhang Xiong. 2022a.

 
 Multi-modal contrastive pre-training for recommendation. In Proceedings of the 2022 International Conference on Multimedia Retrieval . 99–108.

 
 
 

 
 Lu et al . (2023) 
 
Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. 2023.

 
 Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action.

 
 CoRR abs/2312.17172 (2023).

 
 
 

 
 Malitesta et al . (2023) 
 
Daniele Malitesta, Giandomenico Cornacchia, Claudio Pomo, Felice Antonio Merra, Tommaso Di Noia, and Eugenio Di Sciascio. 2023.

 
 Formalizing Multimedia Recommendation through Multimodal Deep Learning.

 
 CoRR abs/2309.05273 (2023).

 
 
 

 
 Mita et al . (2023) 
 
Masato Mita, Soichiro Murakami, Akihiko Kato, and Peinan Zhang. 2023.

 
 CAMERA: A Multimodal Dataset and Benchmark for Ad Text Generation.

 
 CoRR abs/2309.12030 (2023).

 
 
 

 
 Murakami et al . (2023) 
 
Soichiro Murakami, Sho Hoshino, and Peinan Zhang. 2023.

 
 Natural Language Generation for Advertising: A Survey.

 
 
 
 arXiv:2306.12719

 

 
 Ni et al . (2023) 
 
Yongxin Ni, Yu Cheng, Xiangyan Liu, Junchen Fu, Youhua Li, Xiangnan He, Yongfeng Zhang, and Fajie Yuan. 2023.

 
 A Content-Driven Micro-Video Recommendation Dataset at Scale.

 
 CoRR abs/2309.15379 (2023).

 
 
 

 
 OpenAI (2023a) 
 
OpenAI. 2023a.

 
 ChatGPT.

 
 https://chat.openai.com/chat .

 
 
 

 
 OpenAI (2023b) 
 
R OpenAI. 2023b.

 
 Gpt-4 technical report. arxiv 2303.08774.

 
 View in Article 2, 5 (2023).

 
 
 

 
 Oquab et al . (2023) 
 
Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, et al . 2023.

 
 DINOv2: Learning Robust Visual Features without Supervision.

 
 CoRR abs/2304.07193 (2023).

 
 
 

 
 Qin et al . (2022) 
 
Yanjun Qin, Yuchen Fang, Haiyong Luo, Fang Zhao, and Chenxing Wang. 2022.

 
 Next Point-of-Interest Recommendation with Auto-Correlation Enhanced Multi-Modal Transformer Network.

 
 Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) .

 
 
 

 
 Radford et al . (2021) 
 
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al . 2021.

 
 Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML) . PMLR, 8748–8763.

 
 
 

 
 Radford et al . (2018) 
 
Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al . 2018.

 
 Improving language understanding by generative pre-training.

 
 (2018).

 
 
 

 
 Radford et al . (2019) 
 
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al . 2019.

 
 Language models are unsupervised multitask learners.

 
 OpenAI blog 1, 8 (2019), 9.

 
 
 

 
 Raffel et al . (2020) 
 
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020.

 
 Exploring the limits of transfer learning with a unified text-to-text transformer.

 
 Journal of Machine Learning Research 21, 140 (2020), 1–67.

 
 
 

 
 Rajput et al . (2024) 
 
Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al . 2024.

 
 Recommender systems with generative retrieval.

 
 Advances in Neural Information Processing Systems (NeurIPS) 36 (2024).

 
 
 

 
 Ramesh et al . (2022) 
 
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022.

 
 Hierarchical text-conditional image generation with clip latents.

 
 arXiv preprint arXiv:2204.06125 1, 2 (2022), 3.

 
 
 

 
 Ramesh et al . (2021) 
 
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. 2021.

 
 Zero-Shot Text-to-Image Generation. In Proceedings of the 38th International Conference on Machine Learning (ICML) , Vol. 139. 8821–8831.

 
 
 

 
 Rombach et al . (2022) 
 
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022.

 
 High-Resolution Image Synthesis with Latent Diffusion Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) . 10674–10685.

 
 
 

 
 Salemi et al . (2023) 
 
Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. 2023.

 
 LaMP: When Large Language Models Meet Personalization.

 
 CoRR (2023).

 
 
 

 
 Schneider et al . (2019) 
 
Steffen Schneider, Alexei Baevski, Ronan Collobert, and Michael Auli. 2019.

 
 wav2vec: Unsupervised Pre-Training for Speech Recognition. In 20th Annual Conference of the International Speech Communication Association (Interspeech) . 3465–3469.

 
 
 

 
 Shang et al . (2023) 
 
Yu Shang, Chen Gao, Jiansheng Chen, Depeng Jin, Meng Wang, and Yong Li. 2023.

 
 Learning fine-grained user interests for micro-video recommendation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) . 433–442.

 
 
 

 
 Shen et al . (2020) 
 
Tiancheng Shen, Jia Jia, Yan Li, Hanjie Wang, and Bo Chen. 2020.

 
 Enhancing music recommendation with social media content: an attentive multimodal autoencoder approach. In 2020 International Joint Conference on Neural Networks (IJCNN) . IEEE, 1–8.

 
 
 

 
 Shen et al . (2024) 
 
Xiaoteng Shen, Rui Zhang, Xiaoyan Zhao, Jieming Zhu, and Xi Xiao. 2024.

 
 PMG: Personalized Multimodal Generation with Large Language Models. In The ACM Web Conference (WWW) .

 
 
 

 
 Shin et al . (2020) 
 
Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, and Sameer Singh. 2020.

 
 AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) . 4222–4235.

 
 
 

 
 Singh et al . (2022) 
 
Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. 2022.

 
 FLAVA: A Foundational Language And Vision Alignment Model. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) . 15617–15629.

 
 
 

 
 Singh et al . (2023) 
 
Anima Singh, Trung Vu, Raghunandan H. Keshavan, Nikhil Mehta, Xinyang Yi, Lichan Hong, Lukasz Heldt, Li Wei, Ed H. Chi, and Maheswaran Sathiamoorthy. 2023.

 
 Better Generalization with Semantic IDs: A case study in Ranking for Recommendations.

 
 CoRR abs/2306.08121 (2023).

 
 
 

 
 Song et al . (2023a) 
 
Mingyang Song, Haiyun Jiang, Shuming Shi, Songfang Yao, Shilong Lu, Yi Feng, Huafeng Liu, and Liping Jing. 2023a.

 
 Is ChatGPT A Good Keyphrase Generator? A Preliminary Study.

 
 CoRR abs/2303.13001 (2023).

 
 
 

 
 Song et al . (2023b) 
 
Xuemeng Song, Chun Wang, Changchang Sun, Shanshan Feng, Min Zhou, and Liqiang Nie. 2023b.

 
 MM-FRec: Multi-Modal Enhanced Fashion Item Recommendation.

 
 IEEE Transactions on Knowledge and Data Engineering 35 (2023), 10072–10084.

 
 
 

 
 Spijkervet and Burgoyne (2021) 
 
Janne Spijkervet and John Ashley Burgoyne. 2021.

 
 Contrastive Learning of Musical Representations. In Proceedings of the 22nd International Society for Music Information Retrieval Conference (ISMIR) . 673–681.

 
 
 

 
 Su et al . (2019) 
 
Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. 2019.

 
 Vl-bert: Pre-training of generic visual-linguistic representations.

 
 arXiv preprint arXiv:1908.08530 (2019).

 
 
 

 
 Sun et al . (2019) 
 
Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019.

 
 BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM) . 1441–1450.

 
 
 

 
 Sun et al . (2023) 
 
Wenqi Sun, Ruobing Xie, Shuqing Bian, Wayne Xin Zhao, and Jie Zhou. 2023.

 
 Universal Multi-modal Multi-domain Pre-trained Recommendation.

 
 CoRR abs/2311.01831 (2023).

 
 
 

 
 Tao et al . (2020) 
 
Zhulin Tao, Yinwei Wei, Xiang Wang, Xiangnan He, Xianglin Huang, and Tat-Seng Chua. 2020.

 
 Mgat: Multimodal graph attention network for recommendation.

 
 Information Processing Management 57, 5 (2020), 102277.

 
 
 

 
 Touvron et al . (2023a) 
 
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al . 2023a.

 
 Llama: Open and efficient foundation language models.

 
 arXiv preprint arXiv:2302.13971 (2023).

 
 
 

 
 Touvron et al . (2023b) 
 
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al . 2023b.

 
 Llama 2: Open foundation and fine-tuned chat models.

 
 arXiv preprint arXiv:2307.09288 (2023).

 
 
 

 
 Trisedya et al . (2022) 
 
Bayu Distiawan Trisedya, Jianzhong Qi, Wei Wang, and Rui Zhang. 2022.

 
 GCP: Graph Encoder With Content-Planning for Sentence Generation From Knowledge Bases.

 
 IEEE Trans. Pattern Anal. Mach. Intell. 44, 11 (2022), 7521–7533.

 
 
 https://doi.org/10.1109/TPAMI.2021.3118703 

 

 
 Tuo et al . (2023) 
 
Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. 2023.

 
 AnyText: Multilingual Visual Text Generation And Editing.

 
 CoRR abs/2311.03054 (2023).

 
 
 

 
 Van Den Oord et al . (2017) 
 
Aaron Van Den Oord, Oriol Vinyals, et al . 2017.

 
 Neural discrete representation learning.

 
 Advances in Neural Information Processing Systems (NeurIPS) 30 (2017).

 
 
 

 
 Wang et al . (2023b) 
 
Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023b.

 
 Voyager: An Open-Ended Embodied Agent with Large Language Models.

 
 CoRR abs/2305.16291 (2023).

 
 
 

 
 Wang et al . (2023c) 
 
Jinpeng Wang, Ziyun Zeng, Yunxiao Wang, Yuting Wang, Xingyu Lu, Tianxiang Li, Jun Yuan, Rui Zhang, Hai-Tao Zheng, and Shu-Tao Xia. 2023c.

 
 Missrec: Pre-training and transferring multi-modal interest-aware sequence representation for recommendation. In Proceedings of the 31st ACM International Conference on Multimedia (MM) . 6548–6557.

 
 
 

 
 Wang et al . (2024a) 
 
Weimin Wang, Jiawei Liu, Zhijie Lin, Jiangqiao Yan, Shuo Chen, Chetwin Low, Tuyen Hoang, Jie Wu, Jun Hao Liew, Hanshu Yan, Daquan Zhou, and Jiashi Feng. 2024a.

 
 MagicVideo-V2: Multi-Stage High-Aesthetic Video Generation.

 
 CoRR abs/2401.04468 (2024).

 
 
 

 
 Wang et al . (2024b) 
 
Ye Wang, Jiahao Xun, Mingjie Hong, Jieming Zhu, Tao Jin, Wang Lin, Haoyuan Li, Linjun Li, Yan Xia, Zhou Zhao, and Zhenhua Dong. 2024b.

 
 EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration. In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) .

 
 
 

 
 Wang et al . (2023a) 
 
Zhenduo Wang, Yuancheng Tu, Corby Rosset, Nick Craswell, Ming Wu, and Qingyao Ai. 2023a.

 
 Zero-shot Clarifying Question Generation for Conversational Search. In Proceedings of the ACM Web Conference (WWW) . 3288–3298.

 
 
 

 
 Wei et al . (2024a) 
 
Tianxin Wei, Bowen Jin, Ruirui Li, Hansi Zeng, Zhengyang Wang, Jianhui Sun, Qingyu Yin, Hanqing Lu, Suhang Wang, Jingrui He, and Xianfeng Tang. 2024a.

 
 Towards Unified Multi-Modal Personalization: Large Vision-Language Models for Generative Recommendation and Beyond.

 
 CoRR (2024).

 
 
 

 
 Wei et al . (2023a) 
 
Wei Wei, Chao Huang, Lianghao Xia, and Chuxu Zhang. 2023a.

 
 Multi-modal self-supervised learning for recommendation. In Proceedings of the ACM Web Conference 2023 . 790–800.

 
 
 

 
 Wei et al . (2024b) 
 
Wei Wei, Jiabin Tang, Lianghao Xia, Yangqin Jiang, and Chao Huang. 2024b.

 
 PromptMM: Multi-Modal Knowledge Distillation for Recommendation with Prompt-Tuning. In Proceedings of the ACM on Web Conference (WWW) . 3217–3228.

 
 
 

 
 Wei et al . (2023b) 
 
Yinwei Wei, Wenqi Liu, Fan Liu, Xiang Wang, Liqiang Nie, and Tat-Seng Chua. 2023b.

 
 Lightgt: A light graph transformer for multimedia recommendation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) . 1508–1517.

 
 
 

 
 Wei et al . (2021) 
 
Yinwei Wei, Xiang Wang, Qi Li, Liqiang Nie, Yan Li, Xuanping Li, and Tat-Seng Chua. 2021.

 
 Contrastive Learning for Cold-Start Recommendation.

 
 CoRR abs/2107.05315 (2021).

 
 
 

 
 Wei et al . (2020) 
 
Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He, and Tat-Seng Chua. 2020.

 
 Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback. In The 28th ACM International Conference on Multimedia (MM) . 3541–3549.

 
 
 

 
 Wei et al . (2019) 
 
Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He, Richang Hong, and Tat-Seng Chua. 2019.

 
 MMGCN: Multi-modal graph convolution network for personalized recommendation of micro-video. In Proceedings of the 27th ACM International Conference on Multimedia (MM) . 1437–1445.

 
 
 

 
 Wu et al . (2019) 
 
Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang, and Xing Xie. 2019.

 
 Neural news recommendation with multi-head self-attention. In Proceedings of the Conference on Empirical Methods in Natural Language Processing and the International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) . 6389–6394.

 
 
 

 
 Wu et al . (2021) 
 
Chuhan Wu, Fangzhao Wu, Tao Qi, and Yongfeng Huang. 2021.

 
 Empowering news recommendation with pre-trained language models. In Proceedings of the 44th international ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) . 1652–1656.

 
 
 

 
 Wu et al . (2022) 
 
Chuhan Wu, Fangzhao Wu, Tao Qi, Chao Zhang, Yongfeng Huang, and Tong Xu. 2022.

 
 MM-Rec: Visiolinguistic Model Empowered Multimodal News Recommendation. In The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) . 2560–2564.

 
 
 

 
 Xi et al . (2023) 
 
Yunjia Xi, Weiwen Liu, Jianghao Lin, Jieming Zhu, Bo Chen, Ruiming Tang, Weinan Zhang, Rui Zhang, and Yong Yu. 2023.

 
 Towards open-world recommendation with knowledge augmentation from large language models.

 
 arXiv preprint arXiv:2306.10933 (2023).

 
 
 

 
 Xiao et al . (2022a) 
 
Fangxiong Xiao, Lixi Deng, Jingjing Chen, Houye Ji, Xiaorui Yang, Zhuoye Ding, and Bo Long. 2022a.

 
 From Abstract to Details: A Generative Multimodal Fusion Framework for Recommendation. In MM . 258–267.

 
 
 

 
 Xiao et al . (2022b) 
 
Shitao Xiao, Zheng Liu, Yingxia Shao, Tao Di, Bhuvan Middha, Fangzhao Wu, and Xing Xie. 2022b.

 
 Training Large-Scale News Recommenders with Pretrained Language Models in the Loop. In The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) . 4215–4225.

 
 
 

 
 Xu et al . (2024) 
 
Lanling Xu, Junjie Zhang, Bingqian Li, Jinpeng Wang, Mingchen Cai, Wayne Xin Zhao, and Ji-Rong Wen. 2024.

 
 Prompting Large Language Models for Recommender Systems: A Comprehensive Framework and Empirical Analysis.

 
 CoRR abs/2401.04997 (2024).

 
 
 

 
 Xu et al . (2021) 
 
Song Xu, Haoran Li, Peng Yuan, Yujia Wang, Youzheng Wu, Xiaodong He, Ying Liu, and Bowen Zhou. 2021.

 
 K-PLUG: Knowledge-injected Pre-trained Language Model for Natural Language Understanding and Generation in E-Commerce. In Findings of EMNLP . 1–17.

 
 
 

 
 Xun et al . (2021) 
 
Jiahao Xun, Shengyu Zhang, Zhou Zhao, Jieming Zhu, Qi Zhang, Jingjie Li, Xiuqiang He, Xiaofei He, Tat-Seng Chua, and Fei Wu. 2021.

 
 Why do we click: visual impression-aware news recommendation. In Proceedings of the 29th ACM International Conference on Multimedia (MM) . 3881–3890.

 
 
 

 
 Xv et al . (2022) 
 
Guipeng Xv, Si Chen, Chen Lin, Wanxian Guan, Xingyuan Bu, Xubin Li, Hongbo Deng, Jian Xu, and Bo Zheng. 2022.

 
 Visual Encoding and Debiasing for CTR Prediction. In Proceedings of the 31st ACM International Conference on Information Knowledge Management (CIKM) . 4615–4619.

 
 
 

 
 Yang et al . (2022) 
 
Shiquan Yang, Rui Zhang, Sarah M. Erfani, and Jey Han Lau. 2022.

 
 An Interpretable Neuro-Symbolic Reasoning Framework for Task-Oriented Dialogue Generation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL) . 4918–4935.

 
 
 

 
 Yang et al . (2019) 
 
Xiao Yang, Tao Deng, Weihan Tan, Xutian Tao, Junwei Zhang, Shouke Qin, and Zongyao Ding. 2019.

 
 Learning Compositional, Visual and Relational Representations for CTR Prediction in Sponsored Search. In CIKM . 2851–2859.

 
 
 

 
 Yang et al . (2023) 
 
Zhiguang Yang, Lu Wang, Chun Gan, Liufang Sang, and et al. 2023.

 
 Parallel Ranking of Ads and Creatives in Real-Time Advertising Systems.

 
 CoRR (2023).

 
 
 

 
 Yao et al . (2024) 
 
Dong Yao, Jieming Zhu, Jiahao Xun, Shengyu Zhang, Zhou Zhao, Liqun Deng, Wenqiao Zhang, Zhenhua Dong, and Xin Jiang. 2024.

 
 MART: Learning Hierarchical Music Audio Representations with Part-Whole Transformer. In Companion Proceedings of the ACM on Web Conference (WWW) . 967–970.

 
 
 

 
 Yi et al . (2022) 
 
Zixuan Yi, Xi Wang, Iadh Ounis, and Craig Macdonald. 2022.

 
 Multi-modal Graph Contrastive Learning for Micro-video Recommendation.

 
 Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) (2022).

 
 
 

 
 Yu et al . (2022b) 
 
Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. 2022b.

 
 CoCa: Contrastive Captioners are Image-Text Foundation Models.

 
 Trans. Mach. Learn. Res. 2022 (2022).

 
 
 

 
 Yu et al . (2022a) 
 
Licheng Yu, Jun Chen, Animesh Sinha, Mengjiao Wang, Yu Chen, Tamara L. Berg, and Ning Zhang. 2022a.

 
 CommerceMM: Large-Scale Commerce MultiModal Representation Learning with Omni Retrieval. In The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) . 4433–4442.

 
 
 

 
 Yuan et al . (2023) 
 
Zheng Yuan, Fajie Yuan, Yu Song, Youhua Li, Junchen Fu, Fei Yang, Yunzhu Pan, and Yongxin Ni. 2023.

 
 Where to go next for recommender systems? id-vs. modality-based recommender models revisited. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) . 2639–2649.

 
 
 

 
 Zeng et al . (2021) 
 
Mingliang Zeng, Xu Tan, Rui Wang, Zeqian Ju, Tao Qin, and Tie-Yan Liu. 2021.

 
 MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training. In Findings of the Association for Computational Linguistics (ACL) . 791–800.

 
 
 

 
 Zhan et al . (2024) 
 
Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, et al . 2024.

 
 AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling.

 
 CoRR abs/2402.12226 (2024).

 
 
 

 
 Zhang et al . (2023b) 
 
Lingzi Zhang, Xin Zhou, and Zhiqi Shen. 2023b.

 
 Multimodal pre-training framework for sequential recommendation via contrastive learning.

 
 arXiv preprint arXiv:2303.11879 (2023).

 
 
 

 
 Zhang et al . (2021) 
 
Qi Zhang, Jingjie Li, Qinglin Jia, Chuyuan Wang, Jieming Zhu, Zhaowei Wang, and Xiuqiang He. 2021.

 
 UNBERT: User-News Matching BERT for News Recommendation.. In IJCAI , Vol. 21. 3356–3362.

 
 
 

 
 Zhang et al . (2023a) 
 
Yiyuan Zhang, Kaixiong Gong, Kaipeng Zhang, Hongsheng Li, Yu Qiao, Wanli Ouyang, and Xiangyu Yue. 2023a.

 
 Meta-Transformer: A Unified Framework for Multimodal Learning.

 
 CoRR abs/2307.10802 (2023).

 
 
 

 
 Zhang and Wang (2023) 
 
Zizhuo Zhang and Bang Wang. 2023.

 
 Prompt Learning for News Recommendation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) . 227–237.

 
 
 

 
 Zhao et al . (2019) 
 
Guoshuai Zhao, Hao Fu, Ruihua Song, Tetsuya Sakai, Zhongxia Chen, Xing Xie, and Xueming Qian. 2019.

 
 Personalized Reason Generation for Explainable Song Recommendation.

 
 ACM Trans. Intell. Syst. Technol. 10, 4 (2019), 41:1–41:21.

 
 
 

 
 Zhou et al . (2023b) 
 
Hongyu Zhou, Xin Zhou, Zhiwei Zeng, Lingzi Zhang, and Zhiqi Shen. 2023b.

 
 A Comprehensive Survey on Multimodal Recommender Systems: Taxonomy, Evaluation, and Future Directions.

 
 CoRR abs/2302.04473 (2023).

 
 
 

 
 Zhou et al . (2024) 
 
Jianghui Zhou, Ya Gao, Jie Liu, Xuemin Zhao, Zhaohua Yang, Yue Wu, and Lirong Shi. 2024.

 
 GCOF: Self-iterative Text Generation for Copywriting Using Large Language Model.

 
 
 
 arXiv:2402.13667

 

 
 Zhou et al . (2022) 
 
Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. 2022.

 
 Learning to Prompt for Vision-Language Models.

 
 Int. J. Comput. Vis. 130, 9 (2022), 2337–2348.

 
 
 

 
 Zhou et al . (2023a) 
 
Wangchunshu Zhou, Yuchen Eleanor Jiang, Ethan Wilcox, Ryan Cotterell, and Mrinmaya Sachan. 2023a.

 
 Controlled Text Generation with Natural Language Instructions. In Proceedings of International Conference on Machine Learning (ICML) . 42602–42613.

 
 
 

 
 Zhou and Shen (2023) 
 
Xin Zhou and Zhiqi Shen. 2023.

 
 A tale of two graphs: Freezing and denoising graph structures for multimodal recommendation. In Proceedings of the 31st ACM International Conference on Multimedia (MM) . 935–943.

 
 
 

 
 Zhu et al . (2022) 
 
Jieming Zhu, Quanyu Dai, Liangcai Su, Rong Ma, Jinyang Liu, Guohao Cai, Xi Xiao, and Rui Zhang. 2022.

 
 BARS: Towards Open Benchmarking for Recommender Systems. In The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) . 2912–2923.

 
 
 

 
 Zhu et al . (2024) 
 
Jieming Zhu, Xin Zhou, Chuhan Wu, Rui Zhang, and Zhenhua Dong. 2024.

 
 Multimodal Pretraining and Generation for Recommendation: A Tutorial. In Companion Proceedings of the ACM on Web Conference 2024 (WWW) . 1272–1275.

 
 
 

 
 Zhu et al . (2023) 
 
Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. 2023.

 
 TryOnDiffusion: A Tale of Two UNets. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) . 4606–4615.

 
 
 

 
 Zhu et al . (2021) 
 
Yushan Zhu, Huaixiao Zhao, Wen Zhang, Ganqiang Ye, Hui Chen, Ningyu Zhang, and Huajun Chen. 2021.

 
 Knowledge Perceived Multi-modal Pretraining in E-commerce. In ACM Multimedia Conference (MM) . 2744–2752.