A Survey of Mix-based Data Augmentation: Taxonomy, Methods, Applications, and Explainability 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2212.10888v2 [cs.LG] 04 Jun 2024 
 
 

# A Survey of Mix-based Data Augmentation: Taxonomy, Methods, Applications, and Explainability

 DOI:  XXXXXXX.XXXXXXX Price:  15.00 ISBN:  978-1-4503-XXXX-X/18/06 CCS:  Computing methodologies Neural networks CCS:  Computing methodologies Supervised learning by classification CCS:  Computing methodologies Regularization 
 
 
 Chengtai Cao
 
 
 
 email: chengtcao2-c@my.cityu.edu.hk 
 
 Affiliation:  City University of Hong Kong , Hong Kong , China 
 
 , 
 Fan Zhou
 
 
 
 email: fan.zhou@uestc.edu.cn 
 
 Note:  Corresponding author: Fan Zhou
 
 Affiliation:  University of Electronic Science and Technology of China , Chengdu , China 
 
 , 
 Yurou Dai
 
 
 
 email: yuroudai2@um.cityu.edu.hk 
 
 Affiliation:  City University of Hong Kong , Hong Kong , China 
 
 , 
 Jianping Wang
 
 
 
 email: jianwang@cityu.edu.hk 
 
 Affiliation:  City University of Hong Kong , Hong Kong , China 
 
 and 
 Kunpeng Zhang
 
 
 
 email: kpzhang@umd.edu 
 
 Affiliation:  University of Maryland , College Park , USA 
 
 Received  5 June 2009 

 Abstract. 
 
 Data augmentation (DA) is indispensable in modern machine learning and deep neural networks. The basic idea of DA is to construct new training data to improve the model’s generalization by adding slightly disturbed versions of existing data or synthesizing new data. This survey comprehensively reviews a crucial subset of DA techniques, namely Mix-based Data Augmentation (MixDA), which generates novel samples by combining multiple examples. In contrast to traditional DA approaches that operate on single samples or entire datasets, MixDA stands out due to its effectiveness, simplicity, flexibility, computational efficiency, theoretical foundation, and broad applicability. We begin by introducing a novel taxonomy that categorizes MixDA into Mixup-based, Cutmix-based, and mixture approaches based on a hierarchical perspective of the data mixing operation. Subsequently, we provide an in-depth review of various MixDA techniques, focusing on their underlying motivations. Owing to its versatility, MixDA has penetrated a wide range of applications, which we also thoroughly investigate in this survey. Moreover, we delve into the underlying mechanisms of MixDA’s effectiveness by examining its impact on model generalization and calibration while providing insights into the model’s behavior by analyzing the inherent properties of MixDA. Finally, we recapitulate the critical findings and fundamental challenges of current MixDA studies while outlining the potential directions for future works. Different from previous related surveys that focus on DA approaches in specific domains (e.g., computer vision and natural language processing) or only review a limited subset of MixDA studies, we are the first to provide a systematical survey of MixDA, covering its taxonomy, methodology, application, and explainability. Furthermore, we provide promising directions for researchers interested in this exciting area. A curated list of reviewed methods can be found at https://github.com/ChengtaiCao/Awesome-Mix.

 
 
 
 Keywords:  Data augmentation, mix strategies, generalization, machine learning.
 
 

## 1. Introduction

 
 Deep learning (DL) has a transformative impact on diverse domains  ( LeCun et al., 2015 ) due to its ability to learn expressive representation from data. As the complexity of the problems being addressed has increased, the architectures of deep neural networks (DNNs) have become increasingly sophisticated. However, DNNs are notorious for being data-hungry with millions, even billions of parameters (e.g., BERT  ( Kenton and Toutanova, 2019 ) ), making them prone to overfitting.

 
 
 Many innovations have been dedicated to making DNNs more data-efficient by developing improved network architectures. For example, convolutional neural networks (CNNs) undergo a remarkable evolution from AlexNet  ( Krizhevsky et al., 2012 ) to Vision Transformers (ViTs)  ( Dosovitskiy et al., 2020 ) . Besides, various regularization techniques have been proposed to enhance the generalization capability of DNNs. Two notable examples are dropout  ( Srivastava et al., 2014 ) and batch normalization  ( Ioffe and Szegedy, 2015 ) . Dropout randomly zeros out some activations during training to simulate an ensemble of sub-networks and prevent co-adaptation of neurons. On the other hand, batch normalization normalizes the activations by subtracting the batch mean and dividing by the batch standard deviation, which helps to stabilize the training process and improve convergence.

 
 
 Data augmentation (DA), which involves increasing the size and the diversity of training data without explicitly collecting new examples, is commonly employed as a remedy to mitigate overfitting. DA methods aim to expand limited data and extract additional information, enhancing the overall model performance combined with advanced network architectures and existing regularization techniques. For instance, adding random noise into samples, as a simple DA method, can generate numerous new training samples to benefit model robustness. In the context of image data, label-invariant data transformations, such as random-cropping and horizontal-flipping  ( Krizhevsky et al., 2012 ) , can boost model performance and robustness. Similarly, training a model with techniques like random erasing  ( Zhong et al., 2020 ) or Cutout  ( DeVries and Taylor, 2017 ) can improve regularization. In the field of natural language processing (NLP), synonym replacement and random deletion  ( Wei and Zou, 2019 ) are prevailing methods for augmenting textual data. Lastly, generative models, including variational auto-encoders (VAE)  ( Kingma and Welling, 2014 ) , generative adversarial networks (GANs)  ( Goodfellow et al., 2014 ) , and generative pre-trained transformers (GPT)  ( Queiroz Abonizio and Barbon Junior, 2020 ; Yoo et al., 2021 ; Anaby-Tavor et al., 2020 ; Bayer et al., 2023 ) , have gained popularity for DA due to their ability to generate an unlimited number of synthetic yet realistic samples.

 
 
 This survey focuses on a burgeoning subfield of data augmentation – Mix-based Data Augmentation (MixDA), which has aroused considerable research in recent years. In contrast to traditional DA methods operating on a single instance or entire dataset, MixDA creates virtual training data by combining multiple examples, thereby generating a great deal of training data without domain knowledge. For example, Mixup  ( Zhang et al., 2018 ) linearly interpolates input-output pairs from two randomly sampled training examples in a holistic perspective. On the other hand, Cutmix  ( Yun et al., 2019 ) cuts a patch from one image (source image) and then pastes it onto the corresponding region of another image (target image) from a locality point of view. Subsequently, numerous improved versions of MixDA have been proposed, built upon the foundations of Mixup and Cutmix. These methods explore different aspects of data mixing, such as flexible mixing ratios, saliency guidance, and improved divergence, which form the basis for the taxonomy presented in this review. Owing to its versatility, MixDA has been successfully applied to a wide range of tasks, including semi-supervised learning, generative models, graph learning, and contrastive learning. Furthermore, several theoretical studies have been conducted to interpret and analyze MixDA from various perspectives.

 
 
 Importance of MixDA. There are several key reasons why this technology has attracted widespread attention in the research community: (i) Effectiveness : MixDA methods have consistently demonstrated superior performance compared to traditional DA techniques. By combining multiple samples, MixDA introduces a higher level of diversity in the training data, encouraging models to learn more robust and generalizable features; (ii) Simplicity and Flexibility : MixDA methods are relatively simple to implement and can be easily integrated into existing deep learning pipelines. They do not require extensive domain knowledge or complex transformations, making them accessible to a wide range of researchers and practitioners; (iii) Computational Efficiency : Compared to other advanced data augmentation techniques, such as generative model based methods  ( Kingma and Welling, 2014 ; Goodfellow et al., 2014 ; Yoo et al., 2021 ; Anaby-Tavor et al., 2020 ; Bayer et al., 2023 ) , MixDA methods are computationally efficient. They do not require additional training or generation of synthetic samples, making them suitable for resource-constrained environments or large-scale datasets; (iv) Theoretical Foundations : MixDA methods have solid theoretical foundations rooted in the principles of vicinal risk minimization, regularization, and calibration; and (v) Broad Applicability : While MixDA methods have been primarily studied in the context of computer vision, their potential extends to other domains as well, such as speech recognition, natural language processing, and graph-structure data analysis. Given the rapid growth, effectiveness, versatility, and theoretical foundations of MixDA, it is evident that a comprehensive survey of this technique is both timely and necessary. This survey aims to provide a thorough overview of the foundations, methods, applications, and explainability of MixDA. By presenting our findings on the current state of MixDA, its challenges, and promising future research directions, we hope to illuminate the path for further advancements in this field.

 
 
 Related Surveys. We clearly state the differences between our survey and related works to highlight its unique contributions. Several works have reviewed DA techniques  ( Feng et al., 2021a ; Yang et al., 2022c ; Shorten and Khoshgoftaar, 2019 ; Wen et al., 2021b ; Zhao et al., 2021 ; Bayer et al., 2022 ; Li et al., 2022a ) , which are related to our work. However, these reviews primarily focus on applying DA methods in specific domains. For instance, Feng et al. (2021a) and Li et al. (2022a) focus on DA methods in text data. Similarly, several surveys reviewing DA methods in other specific areas, such as image recognition  ( Yang et al., 2022c ; Shorten and Khoshgoftaar, 2019 ) , time series learning  ( Wen et al., 2021b ) , graph learning  ( Zhao et al., 2021 ) and text classification  ( Bayer et al., 2022 ) . Although there is some overlap between these works and our survey, such as Bayer et al. (2022) providing a concise overview of DA methods for text classification, including some MixDA approaches for textual data, our study distinguishes itself by focusing exclusively on MixDA, which can be exploited in various domains (cf. Section  4 for details).

 
 
 Among the related surveys, work  ( Naveed et al., 2024 ) and work  ( Lewy and Mańdziuk, 2022 ) are the most closely related to our survey. The former reviews the methods for image mix and image deletion while the latter reviews both mix augmentation and other augmentation strategies. However, these works only provide a summary of a small subset of MixDA methods and have other focal points: Naveed et al. (2024) review some deleting-based DA methods such as Random Erasing  ( Zhong et al., 2020 ) and Lewy et al.  ( Lewy and Mańdziuk, 2022 ) discuss some cut-based DA techniques such as Cutout  ( DeVries and Taylor, 2017 ) that erases a region of the input image with 0 0 pixel value. Moreover, neither discusses the applications of MixDA across various domains, which is a crucial aspect of MixDA given its versatility. In contrast, our survey dedicates its full attention to MixDA, providing a comprehensive overview of its foundations, methods, and explainability. Most importantly, we also present a thorough review of MixDA applications, which previous studies have not covered. Furthermore, we present findings with respect to (w.r.t.) the current research and provide insight into remaining open challenges, as well as some promising future research directions. To our knowledge, this survey is the first comprehensive work to review MixDA techniques and summarize their wide spectrum of applications. In particular, we review more than 70 70 MixDA methods and more than 8 8 MixDA applications.

 
 
 Organization. This survey is structured as follows. The overall picture of DA and MixDA is depicted in Section  2 , where we also provide a new classification of MixDA. Section  3 systematically reviews existing methods in a more fine-grained taxonomy. Section  4 investigates the important applications of MixDA, followed by the explainability analysis of MixDA in Section  5 . Moreover, the critical findings and challenges are presented in Section  6 , which also outlines the potential research directions. Finally, we conclude this work in Section  7 .

 
 
 

## 2. Preliminary

 

### 2.1. Data Augmentation (DA)

 
 The quantity and diversity of training data play a crucial role in the performance of machine learning models. To harness the full potential of machine learning approaches, numerous DA methods have been proposed to increase training data’s size, variety, and quality by covering unexplored input space while ensuring the correctness of the associated labels. In addition to the differences in inherent augmentation techniques and their applicable domains  ( Feng et al., 2021a ; Yang et al., 2022c ; Shorten and Khoshgoftaar, 2019 ; Wen et al., 2021b ; Zhao et al., 2021 ; Bayer et al., 2022 ; Li et al., 2022a ) , DA methods can be broadly categorized into three classes: (i) single-sample DA (SsDA), (ii) mix-based DA (MixDA), and (iii) generative-based DA (GenDA). Formally, SsDA approaches generate new instances ( 𝐱 ~ i , 𝐲 ~ i ) (\tilde{\mathbf{x}}_{i},\tilde{\mathbf{y}}_{i}) for the training sample ( 𝐱 i , 𝐲 i ) (\mathbf{x}_{i},\mathbf{y}_{i}) by applying transformations to the original sample while preserving its label:

 

 
 (1) | 
 | 
 𝐱 ~ i = 𝒜 θ ​ ( 𝐱 i ) , 𝐲 ~ i = 𝐲 i , \tilde{\mathbf{x}}_{i}=\mathcal{A}_{\mathbf{\theta}}(\mathbf{x}_{i}),\ \tilde{\mathbf{y}}_{i}=\mathbf{y}_{i}, | 
 | 
 

 where 𝒜 θ \mathcal{A}_{\mathbf{\theta}} is a specific feature transformation operation (e.g., rotation) with parameters θ \mathbf{\theta} (e.g., rotation angles). On the contrary, MixDA constructs synthetic data by combining multiple training examples, which can be formulated as:

 

 
 (2) | 
 | 
 𝐱 ~ i = 𝒜 θ ​ ( 𝐱 i , 𝐱 j ) , 𝐲 ~ i = 𝒜 ϕ ​ ( 𝐲 i , 𝐲 j ) , \tilde{\mathbf{x}}_{i}=\mathcal{A}_{\mathbf{\theta}}(\mathbf{x}_{i},\mathbf{x}_{j}),\ \tilde{\mathbf{y}}_{i}=\mathcal{A}_{\mathbf{\phi}}(\mathbf{y}_{i},\mathbf{y}_{j}), | 
 | 
 

 where 𝒜 ϕ \mathcal{A}_{\mathbf{\phi}} denotes the label combination scheme parameterized by ϕ \mathbf{\phi} . It is worth noting that Equation ( 2 ) illustrates the mixing of only two training examples for simplicity. However, extending the formulation to blend more samples is straightforward. Unlike SsDA and MixDA, which generate new data at the sample level, GenDA produces synthetic examples with labels based on the generative model ℳ \mathcal{M} . This model is trained or fine-tuned on the entire dataset 𝒟 \mathcal{D} :

 

 
 (3) | 
 | 
 ℳ = Train ⁡ ( ℳ , 𝒟 ) \displaystyle\mathcal{M}=\operatorname{Train}(\mathcal{M},\mathcal{D}) | 
 or ​ ℳ = Fine − Tune ⁡ ( ℳ , 𝒟 ) , \displaystyle\ \text{or}\ \mathcal{M}=\operatorname{Fine-Tune}(\mathcal{M},\mathcal{D}), | 
 | 
 
 
 (4) | 
 | 
 ( 𝐱 ~ i , 𝐲 ~ i ) \displaystyle(\tilde{\mathbf{x}}_{i},\tilde{\mathbf{y}}_{i}) | 
 = ℳ ⁡ ( ) , \displaystyle=\mathcal{M}(), | 
 | 
 

 
 
 GenDA can be easily distinguished from SsDA and MixDA, as the latter is at sample level. In contrast, GenDA requires training or fine-tuning a generative model on the entire dataset and then using the trained model to synthesize new instances. SsDA and MixDA have three main differences: (i) MixDA creates new training instances by combining multiple training samples, while SsDA only considers a single example; (ii) the feature transformation in SsDA is usually label-invariant, meaning that the newly generated instances have the same target label. In contrast, the label combination strategy 𝒜 ϕ \mathcal{A}_{\mathbf{\phi}} in MixDA should be carefully designed to align with the feature mixing process; and (iii) most MixDA techniques are domain-agnostic and can be applied to various fields, whereas SsDA techniques are often domain-specific. Given the numerous advantages of MixDA, such as its effectiveness, flexibility, theoretical foundations, and broad applicability, we focus on this particular category of DA methods in this survey.

 
 
 

### 2.2. Taxonomy of MixDA

 
 To better understand the landscape of MixDA methods, we propose a novel taxonomy that categorizes existing techniques into three main groups: (i) methods that mix training examples from a global perspective, represented by the pioneering work Mixup  ( Zhang et al., 2018 ) ; (ii) approaches that construct new data through the lens of locality , represented by Cutmix  ( Yun et al., 2019 ) ; and (iii) other techniques that are based on the principle of mixing but cannot be simply grouped into the above two categories, such as mixing with data reconstruction and integrating multiple MixDA solutions. The rationale behind our proposed taxonomy is as follows. Mixup and its variants typically develop and apply a global mix scheme to all features. For instance, Mixup draws a mix ratio from a Beta distribution, and each feature in the created example is a linear combination of the corresponding features from two sampled training examples, encouraging the model to understand data globally . In contrast, Cutmix and its adaptations intercept partial features from one instance and paste them on another, aiming to improve the model’s localization ability. Furthermore, several works exist that integrate multiple MixDA approaches or combine MixDA with other SsDA methods. For example, RandomMix  ( Liu et al., 2022e ) creates augmented data by sampling a mixing operation from a set of MixDA methods for each mini-batch. Similarly, AugMix  ( Hendrycks et al., 2020 ) constructs multiple versions for each sample using SsDA and then mixes them via MixDA techniques.

 
 
 
 

## 3. MixDA Methods

 
 Table 1. The commonly used MixDA benchmarks and learning tasks. 
 
 
 Benchmark | 
 Modality | 
 Task | 
 Article | 

 
 CIFAR  ( Krizhevsky et al., 2009 ) | 
 Image | 
 Image Classification | 
 
 
 
 ( Archambault et al., 2019 ; Chidambaram et al., 2022 ; Guo et al., 2019b ; Dabouei et al., 2021 ; Baek et al., 2021 ; Cascante-Bonilla et al., 2021 ; Greenewald et al., 2023 ; Chen et al., 2022a ; Faramarzi et al., 2022 ; Berthelot et al., 2019c ; Bunk et al., 2021 ; Feng et al., 2021b ; Choi et al., 2022 ) 
 
 ( Harris et al., 2020 ; Hataya and Nakayama, 2022 ; Hendrycks et al., 2020 ; Hendrycks et al., 2022 ; Hong et al., 2021 ; Kim et al., 2021 ; Kim et al., 2020c ; Kim et al., 2020a ; Li et al., 2021c ; Li et al., 2021b ; Liang et al., 2018 ; Liu et al., 2022d ) 
 
 ( Mai et al., 2021 ; Liu et al., 2022a ; Liu et al., 2022b ; Liu et al., 2018 ; Park et al., 2022c ; Mangla et al., 2020 ; Park et al., 2022b ; Pinto et al., 2022 ; Qin et al., 2020 ; Muhammad et al., 2021 ) 
 
 ( Venkataramanan et al., 2022b ; Uddin et al., 2021 ; Venkataramanan et al., 2022a ; Verma et al., 2019a ; Ramé et al., 2021 ; Summers and Dinneen, 2019 ; Sun et al., 2024 ; Tokozume et al., 2018a ; Takahashi et al., 2018 ) 
 
 ( Walawalkar et al., 2020 ; Yun et al., 2019 ; Zhang et al., 2018 ; Zhang et al., 2021a ; Yu et al., 2021 ; Zhang et al., 2022c ; Zhu et al., 2020 ; Yang et al., 2022b ) 
 | 

 
 Model Robustness Analysis | 
 
 
 
 ( Chen et al., 2021b ; Lamb et al., 2019 ; Baena et al., 2022 ; Chou et al., 2020 ; Kim et al., 2020a ; Hendrycks et al., 2022 ; Lee et al., 2020b ; Greenewald et al., 2023 ; Hong et al., 2021 ; Faramarzi et al., 2022 ; Lee et al., 2020a ; Lim et al., 2022 ) 
 
 ( Verma et al., 2019a ; Wen et al., 2021a ; Zhang et al., 2018 ; Liu et al., 2022a ; Pang et al., 2020 ; Venkataramanan et al., 2022b ) 
 | 

 
 Model Uncertainty Analysis | 
 ( Wen et al., 2021a ; Pinto et al., 2022 ) | 

 
 Semi-Supervised Image Classification | 
 ( Mai et al., 2021 ; Ramé et al., 2021 ; Liu et al., 2022a ; Verma et al., 2019b ; Berthelot et al., 2019b ; Li et al., 2020a ; Wei et al., 2020b ; Li et al., 2022b ; Sun et al., 2022 ) | 

 
 ImageNet  ( Russakovsky et al., 2015 ) | 
 Image | 
 Image Classification | 
 
 
 
 ( Harris et al., 2020 ; Guo et al., 2019b ; Hendrycks et al., 2020 ; Hendrycks et al., 2022 ; Dabouei et al., 2021 ; Baek et al., 2021 ; Cascante-Bonilla et al., 2021 ; Chen et al., 2022a ; Hataya and Nakayama, 2022 ; Hong et al., 2021 ; Choi et al., 2022 ; Chen et al., 2022b ; Kim et al., 2021 ) 
 
 ( Li et al., 2021b ; Li et al., 2020b ; Liang et al., 2018 ; Kim et al., 2020a ; Li et al., 2021c ; Liu et al., 2022c ; Liu et al., 2022d ; Liu et al., 2018 ; Lee et al., 2020b ; Liu et al., 2022a ; Liu et al., 2022b ; Mai et al., 2021 ) 
 
 ( Mangla et al., 2020 ; Muhammad et al., 2021 ; Park et al., 2022c ; Park et al., 2022b ; Takahashi et al., 2018 ; Tokozume et al., 2018a ; Uddin et al., 2021 ; Qin et al., 2020 ; Pinto et al., 2022 ; Ramé et al., 2021 ) 
 
 ( Zhang et al., 2018 ; Zhu et al., 2020 ; Venkataramanan et al., 2022a ; Venkataramanan et al., 2022b ; Verma et al., 2019a ; Yang et al., 2022b ; Yu et al., 2021 ; Yun et al., 2019 ) 
 | 

 
 Model Robustness Analysis | 
 ( Zhang et al., 2018 ; Thulasidasan et al., 2019 ; Uddin et al., 2021 ; Chen et al., 2022b ; Kim et al., 2021 ; Venkataramanan et al., 2022a ; Cascante-Bonilla et al., 2021 ; Liu et al., 2022a ; Pinto et al., 2022 ; Lee et al., 2020a ; Lim et al., 2022 ) | 

 
 Model Uncertainty Analysis | 
 ( Thulasidasan et al., 2019 ; Chen et al., 2022b ; Venkataramanan et al., 2022a ; Wen et al., 2021a ; Pinto et al., 2022 ) | 

 
 Object Localization | 
 ( Yun et al., 2019 ; Kim et al., 2021 ; Venkataramanan et al., 2022a ) | 

 
 Pascal VOC Object Detection | 
 
 
 
 ( Uddin et al., 2021 ; Chen et al., 2022b ; Cascante-Bonilla et al., 2021 ; Qin et al., 2020 ; Hataya and Nakayama, 2022 ; Liu et al., 2022b ; Venkataramanan et al., 2022b ; Li et al., 2021c ; Olsson et al., 2021 ; Li et al., 2020b ; Chu et al., 2020 ; Lee et al., 2021b ) 
 | 

 
 MS-COCO Image Captioning | 
 ( Yun et al., 2019 ; Chen et al., 2022b ; Yang et al., 2022b ; Yu et al., 2021 ; Cascante-Bonilla et al., 2021 ; Li et al., 2021b ; Venkataramanan et al., 2022b ; Li et al., 2020b ) | 

 
 MNIST  ( LeCun et al., 1998 ) | 
 Image | 
 Image Classification | 
 ( Guo et al., 2019b ; Chidambaram et al., 2022 ; Mai et al., 2021 ; Zhang et al., 2021a ; Harris et al., 2020 ; Baena et al., 2022 ; Zhu et al., 2020 ; Greenewald et al., 2023 ; Beckham et al., 2019 ; Berthelot et al., 2019c ; Feng et al., 2021b ) | 

 
 Semi-Supervised Image Classification | 
 ( Wei et al., 2020b ; Li et al., 2022b ) | 

 
 CIFAR-C  ( Hendrycks and Dietterich, 2019 ) | 
 Image | 
 Model Robustness Analysis | 
 ( Hendrycks et al., 2022 ; Lee et al., 2020b ; Hendrycks et al., 2020 ; Ramé et al., 2021 ; Ramé et al., 2021 ; Liu et al., 2022b ; Wen et al., 2021a ; Pinto et al., 2022 ; Laugros et al., 2020 ; Lim et al., 2022 ; Ren et al., 2022 ) | 

 
 ImageNet-C  ( Hendrycks and Dietterich, 2019 ) | 
 Image | 
 Model Robustness Analysis | 
 ( Hendrycks et al., 2022 ; Lee et al., 2020b ; Hendrycks et al., 2020 ; Ramé et al., 2021 ; Ramé et al., 2021 ; Liu et al., 2022b ; Wen et al., 2021a ; Pinto et al., 2022 ; Laugros et al., 2020 ; Lim et al., 2022 ; Ren et al., 2022 ) | 

 
 SVHN  ( Netzer et al., 2011 ) | 
 Image | 
 Image Classification | 
 ( Verma et al., 2019a ; Baena et al., 2022 ; Mai et al., 2021 ; Greenewald et al., 2023 ; Ramé et al., 2021 ; Faramarzi et al., 2022 ; Beckham et al., 2019 ; Berthelot et al., 2019c ; Mangla et al., 2020 ; Feng et al., 2021b ) | 

 
 Semi-Supervised Image Classification | 
 ( Mai et al., 2021 ; Verma et al., 2019b ; Berthelot et al., 2019b ; Sun et al., 2022 ) | 

 
 Model Robustness Analysis | 
 ( Lamb et al., 2019 ; Lee et al., 2020a ; Chen et al., 2021b ) | 

 
 CUB200-2011  ( Wah et al., 2011 ) | 
 Image | 
 Object Localization | 
 ( Yun et al., 2019 ; Baek et al., 2021 ; Cascante-Bonilla et al., 2021 ; Li et al., 2020c ; Li et al., 2021b ; Liu et al., 2022b ) | 

 
 Fine-Grained Image Classification | 
 ( Huang et al., 2021 ; Yu et al., 2021 ; Liu et al., 2022b ; Liu et al., 2022a ) | 

 
 iNaturalist  ( Van Horn et al., 2018 ) | 
 Image | 
 Unbalanced Image Classification | 
 ( Chou et al., 2020 ; Li et al., 2021b ; Liu et al., 2022b ) | 

 
 FGVC-Aircraft  ( Maji et al., 2013 ) | 
 Image | 
 Fine-Grained Image Classification | 
 ( Huang et al., 2021 ; Li et al., 2020c ; Li et al., 2021b ; Liu et al., 2022b ; Liu et al., 2022a ) | 

 
 UCI  ( Asuncion and Newman, 2007 ) | 
 Image | 
 Tabular Data Classification | 
 ( Zhang et al., 2018 ; Yu et al., 2021 ; Greenewald et al., 2023 ) | 

 
 MR  ( Maas et al., 2011 ) | 
 Image | 
 Sentence Classification | 
 ( Guo, 2020 ; Liu et al., 2021a ; Guo et al., 2019a ) | 

 
 TREC  ( Pang and Lee, 2005 ) | 
 Image | 
 Sentence Classification | 
 ( Yoon et al., 2021a ; Guo, 2020 ; Liu et al., 2021a ; Kwon and Lee, 2022 ; Sawhney et al., 2022 ; Guo et al., 2019a ; Jindal et al., 2020 ) | 

 
 SST  ( Socher et al., 2013 ) | 
 Text | 
 Sentence Classification | 
 ( Guo, 2020 ; Liu et al., 2021a ; Kwon and Lee, 2022 ; Sawhney et al., 2022 ; Guo et al., 2019a ; Jindal et al., 2020 ) | 

 
 Subj  ( Pang and Lee, 2004 ) | 
 Text | 
 Sentence Classification | 
 ( Guo, 2020 ; Liu et al., 2021a ; Kwon and Lee, 2022 ; Guo et al., 2019a ; Jindal et al., 2020 ) | 

 
 GLUE  ( Wang et al., 2019 ) | 
 Text | 
 Natural Language Understanding | 
 ( Yoon et al., 2021a ; Yin et al., 2021 ; Sun et al., 2020 ; Zhang et al., 2022d ) | 

 
 Google Command  ( Warden, 2018 ) | 
 Audio | 
 Audio Classification | 
 ( Zhang et al., 2018 ; Harris et al., 2020 ; Kim et al., 2021 ; Li et al., 2021c ) | 

 
 
 In this section, we review a wide variety of MixDA strategies, which can be categorized into three groups based on our proposed taxonomy: (i) Mixup  ( Zhang et al., 2018 ) and its variants, (ii) Cutmix  ( Yun et al., 2019 ) and its adaptations, and (iii) other MixDA methods. Specifically, in Section  3.1 , we review Mixup-based methods. We begin by introducing the foundational work, Mixup, and then discuss its adaptations from various perspectives. Moving on to Section  3.2 , we start by presenting the influential work Cutmix. Subsequently, we examine its enhancements from different angles. Then, in Section  3.3 , we review other mix-based methods that do not fit neatly into the two groups above. Finally, Section  3.4 discusses Mixup-based and Cutmix-based approaches, highlighting their strengths and weaknesses. To provide a comprehensive overview, we summarize the commonly used benchmarks and corresponding tasks in Table  1 , the characteristics of Mixup-based methods and Cutmix-based approaches in Table  2 and Table  4 , respectively.

 
 

### 3.1. Mixup-based Methods

 

#### 3.1.1. Mixup

 
 Mixup  ( Zhang et al., 2018 ) is the seminal work in MixDA, proposing a straightforward, data-independent, model-agnostic, effective, and efficient principle to construct new training instances. Mixup imposes an inductive bias on training distribution, assuming that linear interpolations of input pairs will result in convex combinations of their outputs, extending the training data distribution. From a regularization perspective, Mixup encourages the model to behave linearly between training examples. Specifically, the combination operation of Mixup is defined as follows:

 

 
 (5) | 
 | 
 𝐱 ~ = λ ​ 𝐱 i + ( 1 − λ ) ​ 𝐱 j , 𝐲 ~ = λ ​ 𝐲 i + ( 1 − λ ) ​ 𝐲 j , \tilde{\mathbf{x}}=\lambda\mathbf{x}_{i}+(1-\lambda)\mathbf{x}_{j},\ \tilde{\mathbf{y}}=\lambda\mathbf{y}_{i}+(1-\lambda)\mathbf{y}_{j}, | 
 | 
 

 where ( 𝐱 i , 𝐲 i ) (\mathbf{x}_{i},\mathbf{y}_{i}) and ( 𝐱 j , 𝐲 j ) (\mathbf{x}_{j},\mathbf{y}_{j}) are two data points randomly sampled from the original training distribution, and ( 𝐱 ~ , 𝐲 ~ ) (\tilde{\mathbf{x}},\tilde{\mathbf{y}}) is the generated instance. The targets 𝐲 i \mathbf{y}_{i} and 𝐲 j \mathbf{y}_{j} are usually represented as one-hot vectors, and λ ∈ [ 0 , 1 ] \lambda\in[0,1] is a hyperparameter controlling the interpolation strength (also known as the mix ratio), which is typically sampled from a Beta distribution: λ ∼ Beta ⁡ ( α , α ) \lambda\sim\operatorname{Beta}(\alpha,\alpha) , where α ∈ ( 0 , ∞ ) \alpha\in(0,\infty) . Mixup shares similarities with SMOTE  ( Chawla et al., 2002 ) , one of the most popular over-sampling approaches for imbalanced classification, as both methods synthesize new instances by interpolating between two existing instances. However, SMOTE performs interpolation between instances in minority classes and their nearest neighbors, while Mixup operates on randomly sampled instances. This difference arises because SMOTE primarily addresses the class imbalance problem, while Mixup is a general DA method.

 
 
 In practical implementation, Mixup operates on each mini-batch and synthesizes new training data using Equation ( 5 ), introducing a small computation overhead. A sketch of Mixup is shown in Figure  1 , where each pixel in generated images is a linear combination of corresponding pixels from sampled images, using a mixing ratio of λ = 0.25 \lambda=0.25 .

 
 
 Figure 1. The sketch of Mixup. Each pixel in the Mixup-generated image is a convex combination of the corresponding pixels from two randomly sampled images with the mix ratio λ \lambda . 
 
 
 Concurrently with Mixup  ( Zhang et al., 2018 ) , Tokozume et al. (2018b) present a Between-Class (BC) learning strategy that composes synthetic sounds by mixing two origin samples and feeds the virtual data to the model. Unlike Mixup, which explicitly blends the target pairs, BC learning encourages the model to predict the mix ratio λ \lambda . Another critical difference between Mixup and BC learning is that the latter takes into account the properties of sound:

 

 
 (6) | 
 | 
 g = 1 1 + 10 G i − G j 20 ⋅ 1 − λ λ , 𝐱 ~ = g ​ 𝐱 i + ( 1 − g ) ​ 𝐱 j g 2 + ( 1 − g ) 2 , g=\frac{1}{1+10^{\frac{G_{i}-G_{j}}{20}}\cdot\frac{1-\lambda}{\lambda}},\ \tilde{\mathbf{x}}=\frac{g\mathbf{x}_{i}+(1-g)\mathbf{x}_{j}}{\sqrt{g^{2}+(1-g)^{2}}}, | 
 | 
 

 where G i G_{i} and G j G_{j} are the sound pressure levels [dB] of 𝐱 i \mathbf{x}_{i} and 𝐱 j \mathbf{x}_{j} , respectively, and in essence, 𝐱 i : 𝐱 j = λ : ( 1 − λ ) \mathbf{x}_{i}:\mathbf{x}_{j}=\lambda:(1-\lambda) . BC learning increases the diversity of training data and regularizes the feature space by maximizing the ratio of the between-class distance to the within-class variance (i.e., Fisher’s criterion). An extension of BC learning for image data is BC+  ( Tokozume et al., 2018a ) , which builds on the fact that CNNs process image data as waveforms.

 
 
 

#### 3.1.2. Mixing in Embedding Space

 
 Word2Vec, as introduced by Mikolov et al. (2013) , has revealed the intriguing property of linear relationships between word embedding. For instance, the equation "king - man + woman ≈ \approx queen" demonstrates the ability to perform arithmetic calculations on word vectors, yielding semantically meaningful results. This linear compositionality of word embeddings has emerged as a desirable property for embedding spaces. Based on this observation, a natural approach is to linearly combine instances in the embedding space to generate augmented data. This approach leverages the inherent linearity in the embedding space to create new instances that preserve the semantic relationships encoded within the vectors.

 
 
 Manifold Mixup  ( Verma et al., 2019a ) linearly combines the intermediate hidden representations of two inputs to utilize the feature space better, smoothing the decision boundaries at different levels of embedding space and reducing the intra-cluster distance. Like Mixup  ( Zhang et al., 2018 ) , the same linear composite of the corresponding output pair is constructed as the new supervision signal. More specifically, given a deep neural network with multiple layers, Manifold Mixup stochastically selects one layer and mixes the embeddings from two randomly sampled examples. The combined result then proceeds to the output layer. Due to the extra randomness in layer selection, compared with Mixup, the loss function in Manifold Mixup has an additional expectation term for optional layers. Venkataramanan et al. (2022a) interpret Mixup from the deformation perspective and propose AlignMixup to further spatially align two sampled embeddings, enabling consistent interpolation in the feature space. To further smooth the decision boundaries and enhance model robustness, Noisy Feature Mixup (NFM)  ( Lim et al., 2022 ) injects noise when performing convex combinations of input-output pairs in the embedding space. This approach has been shown to achieve a favorable trade-off between performance on clean data and robustness against various types of data attacks.

 
 
 Linear interpolation in the embedding space directly addresses the decision boundary issue and provides greater diversity than mixing solely in the data space. This approach generates augmented data with increased variation, leading to more effective regularization effects.

 
 
 

#### 3.1.3. Adaptive Mix Strategy

 
 Mixup can be viewed as the imposition of "local linearity" constraints outside the data manifold, known as out-of-manifold regularization  ( Guo et al., 2019b ) . One potential limitation with Mixup is that the mix ratio, denoted as λ \lambda , obtained through a blind sampling process, may not be optimal. An original example in the data manifold assigned a soft label may conflict with its actual label. This phenomenon is known as "manifold intrusion."

 
 
 To adaptively generate a mix ratio, AdaMixUp  ( Guo et al., 2019b ) is proposed, which employs an auxiliary network to automatically determine a flexible combination scheme and designs a novel objective function to mitigate manifold intrusion. Mai et al. (2021) apply the meta-learning paradigm to learn to mix , which also aims to address the underfitting issue caused by manifold intrusion. Instead of using a predefined distribution for the mix policy, the introduced MetaMixUp  ( Mai et al., 2021 ) dynamically determines the combination strategy in a data-adaptive manner. Specifically, MetaMixUp is a two-level framework in which a meta model explores a new interpolation scheme that guides the main model. AutoMix  ( Liu et al., 2022b ) decomposes the mix training into two sub-tasks: mixed data generation with a Mix Block and mix classification. It unifies both tasks in an end-to-end framework to optimize the combination policy directly.

 
 
 An unexpected phenomenon of Mixup training is that when combined with an ensemble model, calibration is undermined  ( Wen et al., 2021a ) due to a trade-off between accuracy and calibration. Similar to manifold intrusion, this trade-off is because the soft target of mixed samples introduces an under-confidence issue, which is further aggravated by ensembles. To address this issue, motivated by the fact that some classes are more challenging to identify than others, CAMixup  ( Wen et al., 2021a ) , instead of a policy from a Beta distribution, adjusts the mix ratio for each class based on its confidence and accuracy and only applies Mixup on challenging classes. Specifically, Mixup is not applied to the under-confident classes (where accuracy is higher than confidence) but is used to synthesize new data for the over-confident classes (where confidence is higher than accuracy) to reduce model confidence.

 
 
 The linearity in Mixup constrains the diversity of augmented examples and leads to the manifold intrusion issue due to its simplicity, which can jeopardize the regularization effect. Nonlinear Mixup  ( Guo, 2020 ) is proposed to synthesize the input-output pairs adaptively. Unlike Mixup, which allocates the same mix ratio to each dimension of input, the input mixing strategy of Nonlinear Mixup uses a matrix 𝐌 ∈ ℝ W × H \mathbf{M}\in\mathbb{R}^{W\times H} for the input 𝐱 𝐢 ∈ ℝ W × H \mathbf{x_{i}}\in\mathbb{R}^{W\times H} , where W W and H H are the width and height of images. Each value of 𝐌 \mathbf{M} is independently sampled from the Beta distribution: 𝐱 ~ = 𝐌 ⊙ 𝐱 i + ( 1 − 𝐌 ) ⊙ 𝐱 j \tilde{\mathbf{x}}=\mathbf{M}\odot\mathbf{x}_{i}+(1-\mathbf{M})\odot\mathbf{x}_{j} , where ⊙ \odot denotes element-wise multiplication. Note that Nonlinear Mixup differs from Cutmix (which will be detailed in the next sub-section) since the combination matrix of Cutmix  ( Yun et al., 2019 ) is binary. When mixing target pair, Nonlinear Mixup projects one-hot label vector to k k -dimensional representation 𝐳 ∈ ℝ k \mathbf{z}\in\mathbb{R}^{k} and the resulting matrix with c c rows (number of categories) and k k columns (dimension of label embedding) is then used to model targets. The combined instance 𝐱 ~ \tilde{\mathbf{x}} passes through a Policy Mapping module, enabling the label assignment of mixed 𝐱 ~ \tilde{\mathbf{x}} to be based on the synthetic input.

 
 
 Adversarial Mixing Policy (AMP)  ( Liu et al., 2021a ) constructs new examples with a perturbed mixing policy, which involves three steps: (i) using a random mixing ratio λ \lambda to interpolate input-output pairs to create new data; (ii) imposing a slight adversarial perturbation to λ \lambda to regenerate the mixed feature without changing the mixed target, introducing non-linearity to the model; and (iii) minimizing the loss function on the adjusted data to train the model.

 
 
 Liu et al.  ( Liu et al., 2022a ) identify an over-smoothing problem in Mixup that originates from the aimless mixing ratio and present a decoupled Mixup loss to achieve a good trade-off between discrimination and smoothness. Similarly, to adapt Mixup to class-imbalanced scenarios where the model is excepted to move the decision boundary towards the majority class, Remix  ( Chou et al., 2020 ) is proposed. It disentangles the input and output combination coefficient to balance the generalization of minority and majority classes. The combination manipulation for input in Remix is the same as in Mixup. However, for the target combination, Remix sets a higher weight on the minority class to favor the label to the minority class.

 
 
 In summary, employing a flexible combination strategy can effectively mitigate the manifold intrusion problem and address the associated issues of under-confidence and over-smoothing, thereby maximizing the effectiveness of MixDA.

 
 
 

#### 3.1.4. Sample Selection

 
 When dissimilar samples are mixed in MixDA, the synthesis of out-of-distribution (OOD) samples can harm the model’s performance. OOD samples during training can lead to decreased generalization and increased confusion for the model. To address this issue, careful sample selection is crucial.

 
 
 In the context of semi-supervised named entity recognition (NER), Local Additivity based Data Augmentation (LADA)  ( Chen et al., 2020b ) generates novel examples by combining closed samples. LADA has two variants: Intra-LADA and Inter-LADA. Intra-LADA obtains new sentences by exchanging words within a single sentence and interpolating among these new sentences. Inter-LADA, on the other hand, combines different sentences to construct new data. Additionally, LADA proposes using a weighted mixture of random samples and k k -nearest neighbor (KNN) samples to balance regularization and noise. For supervised neural machine translation, Continuous Semantic Augmentation (CSANMT)  ( Wei et al., 2022 ) generates augmented data for each sample within a semantic adjacency region to cover sufficiently diverse synonymous representations. More specifically, an encoder is trained to assign a semantic neighborhood in the hidden space to each training sample, where tangent points within the neighborhood are semantically equivalent via tangential contrast. CSANMT proposes a Mixed Gaussian Recurrent Chain (MGRC) procedure to select a group of representations from the semantic neighborhood, and each of these is incorporated into the continuous hidden space. In the context of self-supervised learning  ( Liu et al., 2021b ) , Zhang et al. (2022c) propose M-Mix to adaptively construct a succession of hard negatives, which can dynamically select multiple examples to mix and assign different weights based on similarity measurement or a learnable function. The regression problem differs from the classification task since interpolation in the continual target space may lead to arbitrarily incorrect labels. To address this issue, Hwang et al.  ( Hwang and Whang, 2021 ) present RegMixup that learns a policy to determine which examples to mix based on the distance metrics using reinforcement learning.

 
 
 Several other approaches share this motivation, such as decaying loss based on the distance between inputs (e.g., Local Mixup  ( Baena et al., 2022 ) ), selecting similar inputs for combination (e.g., Pani  ( Sun et al., 2024 ) ), interpolating in hyperbolic space (e.g., HypMix  ( Sawhney et al., 2021 ) and DMix  ( Sawhney et al., 2022 ) ), employing local-emphasized and global-constrained sub-tasks (e.g., SAMix  ( Li et al., 2021b ) ), and utilizing generative models (e.g., GenLabel  ( Sohn et al., 2022 ) ).

 
 
 In conclusion, the adverse effects of OOD data in MixDA can be minimized by employing appropriate sample selection schemes during the mixing process. By carefully choosing samples that are more similar or share common characteristics, the synthesis of virtual examples through mixing becomes more plausible.

 
 
 

#### 3.1.5. Saliency Style Guidance

 
 As discussed by  Huang and Mumford (1999) , saliency information, which refers to the prominence or importance of specific features or objects within data, is informative and exhibits regularity in various contexts. This regularity in saliency information suggests that it can be leveraged to guide the Mixup process in data augmentation, potentially improving the effectiveness of the augmented instances.

 
 
 SuperMix  ( Dabouei et al., 2021 ) , a supervised interpolation policy, is introduced to exploit the semantics of inputs for generating new training instances. SuperMix incorporates two additional loss functions: one for smoothing the mask matrices and the other for encouraging the sparsity of masks. StyleMix  ( Hong et al., 2021 ) further distinguishes content features from style characteristics when mixing two input images, aiming to improve the variety of synthetic examples. Specifically, let 𝒢 \mathcal{G} and ℋ \mathcal{H} denote the pre-trained style encoder and style decoder, respectively. StyleMix obtains four features:

 

 
 (7) | 
 | 
 𝐟 i ​ i = 𝒢 ⁡ ( 𝐱 i ) , 𝐟 j ​ j = 𝒢 ⁡ ( 𝐱 j ) , 𝐟 i ​ j = AdaIN ⁡ ( 𝐟 i ​ i , 𝐟 j ​ j ) , 𝐟 j ​ i = AdaIN ⁡ ( 𝐟 j ​ j , 𝐟 i ​ i ) , \mathbf{f}_{ii}=\mathcal{G}(\mathbf{x}_{i}),\ \mathbf{f}_{jj}=\mathcal{G}(\mathbf{x}_{j}),\ \mathbf{f}_{ij}=\operatorname{AdaIN}(\mathbf{f}_{ii},\mathbf{f}_{jj}),\ \mathbf{f}_{ji}=\operatorname{AdaIN}(\mathbf{f}_{jj},\mathbf{f}_{ii}), | 
 | 
 

 where AdaIN \operatorname{AdaIN}   ( Huang and Belongie, 2017 ) adaptively normalizes features with the mean 𝝁 ⁡ ( ⋅ ) \bm{\mu}(\cdot) and the standard deviation 𝝈 ⁡ ( ⋅ ) \bm{\sigma}(\cdot) :

 

 
 (8) | 
 | 
 AdaIN ⁡ ( 𝐟 i ​ i , 𝐟 j ​ j ) = 𝝈 ⁡ ( 𝐟 j ​ j ) ​ ( 𝐟 i ​ i − 𝝁 ⁡ ( 𝐟 i ​ i ) 𝝈 ⁡ ( 𝐟 i ​ i ) ) + 𝝁 ⁡ ( 𝐟 j ​ j ) . \operatorname{AdaIN}(\mathbf{f}_{ii},\mathbf{f}_{jj})=\bm{\sigma}(\mathbf{f}_{jj})(\frac{\mathbf{f}_{ii}-\bm{\mu}(\mathbf{f}_{ii})}{\bm{\sigma}(\mathbf{f}_{ii})})+\bm{\mu}(\mathbf{f}_{jj}). | 
 | 
 

 The mixed image is then generated by interpolating the four features:

 

 
 (9) | 
 | 
 𝐱 ~ = ℋ ⁡ ( t ​ 𝐟 i ​ i + ( 1 − λ t − λ s + t ) ​ 𝐟 j ​ j + ( λ t − t ) ​ 𝐟 i ​ j + ( λ s − t ) ​ 𝐟 j ​ i ) , \tilde{\mathbf{x}}=\mathcal{H}(t\mathbf{f}_{ii}+(1-\lambda_{t}-\lambda_{s}+t)\mathbf{f}_{jj}+(\lambda_{t}-t)\mathbf{f}_{ij}+(\lambda_{s}-t)\mathbf{f}_{ji}), | 
 | 
 

 where λ t \lambda_{t} and λ s \lambda_{s} are the content and style mix ratios drawn from the Beta distribution. Hyperparameter t t is constrained by max ⁡ ( 0 , λ t + λ s − 1 ) ≤ t ≤ min ⁡ ( λ t , λ s ) \max(0,\lambda_{t}+\lambda_{s}-1)\leq t\leq\min(\lambda_{t},\lambda_{s}) and the combined target of the mixed image is:

 

 
 (10) | 
 | 
 𝐲 c = λ t ​ 𝐲 i + ( 1 − λ t ) ​ 𝐲 j , 𝐲 s = λ s ​ 𝐲 i + ( 1 − λ s ) ​ 𝐲 j , 𝐲 ~ = λ ​ 𝐲 c + ( 1 − λ ) ​ 𝐲 s . \mathbf{y}_{c}=\lambda_{t}\mathbf{y}_{i}+(1-\lambda_{t})\mathbf{y}_{j},\ \mathbf{y}_{s}=\lambda_{s}\mathbf{y}_{i}+(1-\lambda_{s})\mathbf{y}_{j},\ \tilde{\mathbf{y}}=\lambda\mathbf{y}_{c}+(1-\lambda)\mathbf{y}_{s}. | 
 | 
 

 
 
 Inspired by the observation that the visual domain is associated with image style (e.g., abstract painting vs. landscape), Mixstyle  ( Zhou et al., 2021 ) probabilistically combines instance-level feature statistics of training data. Due to the modular nature of CNNs, style information is modeled in the bottom layers, where the proposed style-mixing operation occurs. Combining styles of the training data may implicitly synthesize new domains, enhancing the domain diversity and further the generalization of the model. Specifically, MixStyle follows the paradigm of mini-batch training, i.e., shuffling and mingling procedures are conducted along the batch dimension:

 

 
 (11) | 
 | 
 γ mix = λ ​ 𝝈 ​ ( 𝐱 ) + ( 1 − λ ) ​ 𝝈 ​ ( 𝐱 ^ ) , β mix = λ ​ 𝝁 ​ ( 𝐱 ) + ( 1 − λ ) ​ 𝝁 ​ ( 𝐱 ^ ) , MixStyle ⁡ ( 𝐱 ~ ) = γ mix ​ 𝐱 − 𝝁 ⁡ ( 𝐱 ) 𝝈 ⁡ ( 𝐱 ) + β mix , \gamma_{\text{mix}}=\lambda\bm{\sigma}(\mathbf{x})+(1-\lambda)\bm{\sigma}(\hat{\mathbf{x}}),\ \beta_{\text{mix}}=\lambda\bm{\mu}(\mathbf{x})+(1-\lambda)\bm{\mu}(\hat{\mathbf{x}}),\ \operatorname{MixStyle}(\tilde{\mathbf{x}})=\gamma_{\text{mix}}\frac{\mathbf{x}-\bm{\mu}(\mathbf{x})}{\bm{\sigma}(\mathbf{x})}+\beta_{\text{mix}}, | 
 | 
 

 where 𝐱 ^ \hat{\mathbf{x}} is the batch reference representation obtained by shuffling the sampled mini-batch from two domains. The mean vector 𝝁 ⁡ ( ⋅ ) \bm{\mu}(\cdot) and standard deviation vectors 𝝈 ⁡ ( ⋅ ) \bm{\sigma}(\cdot) are calculated across the spatial dimension within each channel of each instance, similar to batch normalization  ( Ioffe and Szegedy, 2015 ) .

 
 
 Huang and Belongie (2017) demonstrate that positional moments (i.e., mean 𝝁 \bm{\mu} and standard deviation 𝝈 \bm{\sigma} ) and instance moments can approximately represent shape and style information, which are usually discarded to accelerate and stabilize training. In contrast, Moment Exchange (MoEx)  ( Li et al., 2021c ) uses this instructive information to generate new samples. To be specific, MoEx switches the moments of an example with those of the other:

 

 
 (12) | 
 | 
 𝐡 i ( j ) = 𝝈 j ​ 𝐡 i − 𝝁 i 𝝈 i + 𝝁 j , \mathbf{h}_{i}^{(j)}=\bm{\sigma}_{j}\frac{\mathbf{h}_{i}-\bm{\mu}_{i}}{\bm{\sigma}_{i}}+\bm{\mu}_{j}, | 
 | 
 

 where 𝐡 i ( j ) \mathbf{h}_{i}^{(j)} is the resulting feature for the i i -th input’s representation 𝐡 i \mathbf{h}_{i} mixed with the j j -th input’s representation 𝐡 j \mathbf{h}_{j} . This combined feature then passes through the network to calculate loss w.r.t. labels 𝐲 i \mathbf{y}_{i} and 𝐲 j \mathbf{y}_{j} .

 
 
 For natural language understanding tasks, Park and Caragea (2022) propose a combination scheme to calibrate a pre-trained language model based on saliency information and the Area Under the Margin (AUM) statistic. Training data is first divided into two groups – a set of samples that are easy to distinguish and a batch of ambiguous samples that are difficult to learn by computing the AUM of each instance. Then, a Mixup process is executed to blend these two sets based on the saliency signal proxied by the gradient norm, reconciling the learning complexity and sample confidence for model calibration. Similarly, explainable artificial intelligence (XAI)  ( Kwon and Lee, 2022 ) explicitly extracts the weight of manipulated words and calculates the mixed label based on their importance. Furthermore, to circumvent the intense computation in saliency detectors, TokenMixup  ( Choi et al., 2022 ) leverages attention map to guide token-level data augmentation by optimally matching sample pairs, achieving a 15 × 15\times speed-up. SciMix  ( Sun et al., 2022 ) is proposed to generate new samples with many characteristics not from their semantic parents by teaching a StyleGAN generator to replace global semantic content from other samples into image backgrounds.

 
 
 Saliency and style information can be leveraged to deduce a content-aware combination policy, providing guidance for the mix process and generating more reasonable augmented data.

 
 
 

#### 3.1.6. Diversity in Mixup

 
 Mixup has specific constraints that can limit the diversity of the augmented data. The primary constraints are linear interpolation within the same batch and blending only two examples.

 
 
 BatchMixup  ( Yin et al., 2021 ) breaks the same-batch rule by interpolating all mini-batches, enlarging the space of Mixup-ed examples. K K -Mixup  ( Greenewald et al., 2023 ) produces more augmented data through perturbing K K instances in the direction of other K K samples and interpolates under the Wasserstein metric. Similarly, Jeong et al. (2021) present a novel K K -image mixing paradigm based on the stick-breaking process under Dirichlet prior. To interpolate an arbitrary number of samples, MultiMix  ( Venkataramanan et al., 2022b ) assigns one vector from a Dirichlet distribution to each sample and implements the mix operation in the network’s last layer. Instead of single output in Mixup, MixMo  ( Ramé et al., 2021 ) is presented with multi-input multi-output sub-modules for combining multiple samples. Specifically, inputs are passed through shared encoding layers and then mingled. The mixed representations are fed to multiple output blocks corresponding to multiple labels. To further enlarge the space of augmented data, Hendrycks et al. (2022) present PixMix to combine origin training data with structurally complex images, which include fractals from DeviantArt 1 1 
 1 
 
 
 
 https://www.deviantart.com and feature visualizations OpenAI Microscope 2 2 
 2 
 
 
 
 https://openai.com/blog/microscope .

 
 
 To summarize, expanding the range of samples to mix and combining more examples during the convex interpolation process are effective strategies to enhance the variety of the constructed data in Mixup-based data augmentation.

 
 
 

#### 3.1.7. Miscellaneous Mixup Methods

 
 In addition to the previously described methods, there are other approaches to improve Mixup from other perspectives.

 
 
 To obtain better robustness , a guided interpolation framework (GIF)  ( Chen et al., 2021b ) is developed to leverage the metainformation from the previous epochs for interpolation guidance. Compared with the conventional Mixup, GIF generates more attackable data (i.e., λ = 0.5 \lambda=0.5 ). Besides, GIF tranquilizes the linear behavior between classes, which benefits standard training but is not favorable to adversarial training for robustness, encouraging the model to predict invariably in a cluster. Through the lens of the exploration-exploitation dilemma in reinforcement learning, Mixup constantly explores the data space. To achieve a pleasurable trade-off between exploration and exploitation, Mixup Without hesitation (MWh)  ( Yu et al., 2021 ) is proposed to divide a set of mini-batch examples into 3 3 parts: (i) the standard Mixup, (ii) the combination of Mixup and basic data augmentation, and (iii) a damped Mixup. Mixup has been proved to be equivalent to discovering Euclidean barycenters to boost the performance of the classifier  ( Zhu et al., 2020 ) . Additionally, motivated by the connection between cross-entropy loss and Kullback–Leibler (KL) divergence, AutoMix  ( Zhu et al., 2020 ) constructs barycenter samples with a generative model where OptTransMix utilizes Wasserstein distance instead of the Euclidean metric to define the mixing policy with the optimal transport-based transformation technique. RegMixup  ( Pinto et al., 2022 ) trains the model on data distribution, which combines the original data and the Mixup vicinal approximation to the data distribution instead of using any of them as the exclusive training objective as in standard empirical risk minimization or original Mixup. This additional Mixup regularization term to the standard loss function provides improved accuracy and uncertainty estimation when evaluating models under OOD data and numerous covariate shifts. Though original Mixup tends to obtain high-entropy models, it cannot separate in-distribution instances from out-distribution ones.

 
 
 Table 2. Summary of Mixup-based DA methods 
 
 
 Adaptation | 
 Method | 
 Characteristic | 

 
 Mixing in Embedding Space | 
 Manifold Mixup  ( Verma et al., 2019a ) | 
 Mix hidden representations | 

 
 AlignMixup  ( Venkataramanan et al., 2022a ) | 
 Spatially align embeddings when mixing | 

 
 NFM  ( Lim et al., 2022 ) | 
 Inject noise in embedding space | 

 
 Adaptive Mix Strategy | 
 AdaMixUp  ( Guo et al., 2019b ) | 
 Auxiliary network determines combination | 

 
 MetaMixUp  ( Mai et al., 2021 ) | 
 Meta model explores interpolation scheme | 

 
 AutoMix  ( Liu et al., 2022b ) | 
 Decompose mix training into sub-tasks | 

 
 CAMixup  ( Wen et al., 2021a ) | 
 Adjust mix ratio based on class confidence and accuracy | 

 
 Nonlinear Mixup  ( Guo, 2020 ) | 
 Input mix strategy is a matrix | 

 
 AMP  ( Liu et al., 2021a ) | 
 Adversarial perturbation to feature mixing | 

 
 Decoupled Mixup  ( Liu et al., 2022a ) | 
 Decouple Mixup loss to discrimination and smoothness | 

 
 Remix  ( Chou et al., 2020 ) | 
 Disentangle input and output combination coefficient | 

 
 Sample Selction | 
 LADA  ( Chen et al., 2020b ) | 
 Mix with closed samples | 

 
 CSANMT  ( Wei et al., 2022 ) | 
 Mix with semantic neighborhood | 

 
 M-Mix  ( Zhang et al., 2022c ) | 
 Dynamically mix based on similarity | 

 
 RegMixup  ( Hwang and Whang, 2021 ) | 
 Learn policy to select examples to mix | 

 
 Local Mixup  ( Baena et al., 2022 ) | 
 Loss decayed with input distance | 

 
 Pani  ( Sun et al., 2024 ) | 
 Mix with similar samples | 

 
 HypMix  ( Sawhney et al., 2021 ) | 
 Interpolate in hyperbolic space | 

 
 DMix  ( Sawhney et al., 2022 ) | 
 Select samples based on embedding diversity | 

 
 SAMix  ( Li et al., 2021b ) | 
 Decompose Mixup into local and global sub-tasks | 

 
 GenLabel  ( Sohn et al., 2022 ) | 
 Learn class-conditional data distribution | 

 
 Saliency Style Guidance | 
 SuperMix  ( Dabouei et al., 2021 ) | 
 Exploit input semantics | 

 
 StyleMix  ( Hong et al., 2021 ) | 
 Discern content and style when mixing | 

 
 Mixstyle  ( Zhou et al., 2021 ) | 
 Combine instance-level feature statistics | 

 
 MoEx  ( Li et al., 2021c ) | 
 Leverage positional moments to generate samples | 

 
 TokenMixup  ( Choi et al., 2022 ) | 
 Attention map guides token-level augmentation | 

 
 SciMix  ( Sun et al., 2022 ) | 
 Replace semantic content into backgrounds | 

 
 Diversity | 
 BatchMixup  ( Yin et al., 2021 ) | 
 Interpolate across mini-batches | 

 
 K K -Mixup  ( Greenewald et al., 2023 ) | 
 Interpolate K K instances with other K K samples | 

 
 MultiMix  ( Venkataramanan et al., 2022b ) | 
 Mix scheme from Dirichlet distribution | 

 
 MixMo  ( Ramé et al., 2021 ) | 
 Multi-input multi-output modules combine samples | 

 
 PixMix  ( Hendrycks et al., 2022 ) | 
 Combine training data with complex images | 

 
 Others | 
 GIF  ( Chen et al., 2021b ) | 
 Leverage previous epochs for interpolation | 

 
 MWh  ( Yu et al., 2021 ) | 
 Balance exploration and exploitation | 

 
 AutoMix  ( Zhu et al., 2020 ) | 
 Wasserstein distance defines mix policy | 

 
 RegMixup  ( Pinto et al., 2022 ) | 
 Combine original data and Mixup approximation | 

 
 
 
 

### 3.2. Cutmix-based Methods

 

#### 3.2.1. Cutmix

 
 Inspired by Cutout  ( DeVries and Taylor, 2017 ) , Random Erasing  ( Zhong et al., 2020 ) , and Cutoff  ( Shen et al., 2020 ) that respectively randomly cut out a region (e.g., a rectangle) of the input images, replace the pixels of the selected area with random values, and remove part of the information within an input sentence, Yun et al. (2019) design the Cutmix scheme which replaces a region of one image with the corresponding patch of the other sample, and blends their targets proportionally to the area of the combined examples. Compared with Cutout, Random Erasing, and Cutoff, each dimension of the Cutmix generated sample is informative, accelerating the training process. Besides, the features from the other sample upgrade the localization ability of the model by encouraging the model to identify the feature from a partial perspective. Specifically, Cutmix constructs new training samples by:

 

 
 (13) | 
 | 
 𝐱 ~ = 𝐌 ⊙ 𝐱 i + ( 1 − 𝐌 ) ⊙ 𝐱 j , 𝐲 ~ = λ ​ 𝐲 i + ( 1 − λ ) ​ 𝐲 j , \tilde{\mathbf{x}}=\mathbf{M}\odot\mathbf{x}_{i}+(1-\mathbf{M})\odot\mathbf{x}_{j},\ \tilde{\mathbf{y}}=\lambda\mathbf{y}_{i}+(1-\lambda)\mathbf{y}_{j}, | 
 | 
 

 where 𝐌 \mathbf{M} is a binary mask indicating the cut-and-paste area of the images. As with Mixup  ( Zhang et al., 2018 ) , the mix ratio λ \lambda is drawn from a Beta distribution. To obtain the binary mask 𝐌 \mathbf{M} , Cutmix generates a bounding box by uniformly sampling a rectangle with height λ ​ H \sqrt{\lambda}H and width λ ​ W \sqrt{\lambda}W . The binary matrix is determined by setting the element within the bounding box as 1 and 0 otherwise. A sketch of Cutmix is shown in Figure  2 , where the upper right 1 / 4 1/4 (i.e., λ = 0.25 \lambda=0.25 ) patch of the dog image is cut and then pasted onto the corresponding region of the cat image.

 
 
 Figure 2. An example of Cutmix with mix ratio λ = 0.25 \lambda=0.25 , in which 1 / 4 1/4 area of the dog image (upper right) is cut and pasted onto the corresponding location of the cat (upper right). 
 
 
 In parallel with Cutmix, work  ( Summers and Dinneen, 2019 ) expands the space of mixed examples and investigates a dozen cut-and-paste policies. Taking Vertical Concat as an example, it vertically integrates the top λ \lambda part of image 𝐱 i \mathbf{x}_{i} with the bottom ( 1 − λ ) (1-\lambda) region of image 𝐱 j \mathbf{x}_{j} . Not limited to only combining two instances, random image cropping and patching (RICAP) blends 4 4 images  ( Takahashi et al., 2018 ) . RICAP draws two parameters a a and b b from the uniform distribution: a ∼ Unif ⁡ ( 0 , W ) , b ∼ Unif ⁡ ( 0 , H ) a\sim\operatorname{Unif}(0,W),b\sim\operatorname{Unif}(0,H) and the cropping size ( w i , h i ) (w_{i},h_{i}) for i i -th image is automatically obtained via w 1 = w 3 = a w_{1}=w_{3}=a , w 2 = w 4 = W − a w_{2}=w_{4}=W-a , h 1 = h 2 = b h_{1}=h_{2}=b , and h 1 = h 4 = H − b h_{1}=h_{4}=H-b . For i i -th image, the coordinate of the upper left corner [ m i , n i ] [m_{i},n_{i}] of the cropped areas is decided by m i ∼ Unif ⁡ ( 0 , W − w i ) m_{i}\sim\operatorname{Unif}(0,W-w_{i}) and n i ∼ Unif ⁡ ( 0 , H − h i ) n_{i}\sim\operatorname{Unif}(0,H-h_{i}) . Similar to Cutmix, the mingled target is defined as mixing their one-hot vectors with the proportional to their areas in the generated image.

 
 
 

#### 3.2.2. Integration with Saliency Information

 
 Figure 3. Illustration of why Cutmix is problematic when the cut-and-paste patch contains no information about the dog. However, based on the label combination policy, the probability of the dog for the new image is non-zero ( 0.25 0.25 ), misleading the learned model. 
 
 
 Cutmix may be problematic since it aimlessly cuts a patch from a source image and pastes it onto the other image while the Cutmix generated label is proportional to the patch area. It is evident that if the cut-and-paste patch is uninformative for the source image, Cutmix would not improve performance but introduce noise to the target image, leading to unstable training due to the biased constructed target. An example is shown in Figure  3 in which the mix ratio λ = 0.25 \lambda=0.25 and the cut patch is on the bottom left, which does not contain any information about the dog. However, the probability value for the dog in the constructed label is 0.25 0.25 , forcing the model to assign a softmax score of 0.25 0.25 to the dog, which is undesirable.

 
 
 To overcome the abovementioned issue, Walawalkar et al. (2020) put forward Attentive Cutmix, which selects the most illustrative regions based on the feature attention map and cuts the most representative parts. Similarly, FocusMix  ( Kim et al., 2020c ) adopts proper sample approaches to cut-and-paste the instructive regions. Motivated by ViTs  ( Dosovitskiy et al., 2020 ) , TransMix  ( Chen et al., 2022b ) is introduced to mix targets of input pair based on the attention maps learned with ViTs and assign larger values for input images with greater attention. However, directly applying Cutmix on ViTs may lead to a token fluctuation phenomenon, i.e., the contributions of input tokens fluctuate as forward propagation, resulting in a different mix ratio in the output tokens. To handle this issue, token-label alignment  ( Xiao et al., 2023 ) is used to track the correspondence between the original and mixed tokens to keep the label for each token by reusing the computed attention at each layer.

 
 
 SaliencyMix  ( Uddin et al., 2021 ) prefers to cut the indicative regions of the source image with the help of the saliency detector and paste it onto the target image. Four accepted saliency detection algorithms are evaluated, and VSFs  ( Montabone and Soto, 2010 ) performs best on both benchmarks. To explore the effect of the combination strategy, five possible schemes (source to target) are investigated: (i) Salient to Corresponding , (ii) Salient to Salient , (iii) Salient to Non-Salient , (iv) Non-Salient to Salient , and (v) Non-Salient to Non-Salient in which (Non-)Salient denotes the (non-)salient regions and Corresponding indicates the same location. Experiment evaluations show that scheme { iii } \{\textbf{iii}\} performs best since it can generate a variety of mixed examples compared with scheme { i } \{\text{i}\} and contains more saliency information compared to schemes { ii , iv , v } \{\text{ii},\text{iv},\text{v}\} .

 
 
 Similar to Salient to Non-Salient strategy in SaliencyMix, Kim et al. (2020a) put forward Puzzle Mix, which jointly hunts for (i) the optimal mask and (ii) the optimal transport by taking advantage of the saliency information and the underlying statistics of images. There are two main differences between SaliencyMix and Puzzle Mix: (i) the determination of the Non-Salient region in Puzzle Mix is formalized as an optimization problem rather than stochastically selecting non-informative regions in SaliencyMix and (ii) the saliency information in Puzzle Mix is the ℓ 2 \ell_{2} norm of the gradient values across input channels rather than based on some detection models in SaliencyMix.

 
 
 Figure 4. Illustration of SSMix. The j j -th sentence’s most irrelevant tokens are replaced with the i i -th sentence’s most related tokens. 
 
 
 To utilize saliency information when mixing two sentences, SSMix  ( Yoon et al., 2021a ) produces a new instance while maintaining the locality of the original input through span-based combination and selecting the tokens most related to the prediction based on the magnitude of a gradient. Figure  4 illustrates SSMix, where the most 20 % 20\% ( λ \lambda ) irrelevant tokens in the j j -th sentence are replaced with the most 20 % 20\% related tokens in the i i -th sentence. Semantically Proportional Mixing (SnapMix)  ( Huang et al., 2021 ) calculates the underlying composition of generated images via class activation map (CAM) to reduce label noise. For the fine-grained recognition tasks, Li et al. (2020c) propose Attribute Mix to exploit semantic information from the input pair at the attribute level. To avoid object information missing and inappropriate mixed labels, ResizeMix  ( Qin et al., 2020 ) directly shrinks one image into a small patch and pastes it on a location of the other input stochastically.

 
 
 In total, saliency information plays a crucial role in Cutmix-based methods due to its significance in determining the locality and label combination scheme. Without considering saliency information, a random or aimless cut-and-paste process during Cutmix could introduce noise into the constructed features and bias into the generated targets. Therefore, leveraging saliency information can help maximize the benefits of training on Cutmix-ed data.

 
 
 Table 3. Summary of selected Cutmix-based Adaptations. 
 
 
 Method | 
 Generated Image | 
 Saliency Map | 

 
 SaliencyMix  ( Uddin et al., 2021 ) | 
 
 
 
 | 
 
 
 
 | 

 
 Puzzle Mix  ( Kim et al., 2020a ) | 
 
 
 
 | 
 
 
 
 | 

 
 ResizeMix  ( Qin et al., 2020 ) | 
 
 
 
 | 
 
 
 
 | 

 
 Saliency Grafting  ( Park et al., 2022b ) | 
 
 
 
 | 
 
 
 
 | 

 
 
 

#### 3.2.3. Improved Divergence

 
 While saliency-guided methods in Cutmix facilitate preserving essential features and structures, there is a potential drawback related to the diversity of the augmented data. Saliency-guided methods often prioritize specific regions deemed informative. This prioritization results in a concentration of augmentation effects on these selected areas, reducing the diversity of the augmented data. Other regions or less salient features may not receive equal augmentation or attention, which could result in a biased representation of the data distribution.

 
 
 To get a better trade-off between plausibility and diversity , Saliency Grafting  ( Park et al., 2022b ) is designed to synthesize diverse and reasonable examples. Instead of choosing the most informative patch, Saliency Grafting scales and truncates the saliency map to increase the number of options and diversity. Additionally, a Bernoulli distribution is used to sample these candidate regions. Saliency is also utilized to guide the target combination and guarantee the rationality of virtual input-output pairs. Similarly, Co-Mixup  ( Kim et al., 2021 ) generates mixed data by maximizing the saliency measure of each mixed sample and the supermodular diversity among the augmented data. Representative Cutmix-based adaptations are summarized in Table  3 , where the generated images with their saliency map are instantiated. To integrate with the accumulated knowledge from the previous iterations, RecursiveMix  ( Yang et al., 2022b ) proposes a recursive mix policy that leverages the historical Input - Prediction - Output triplets. Specifically, RecursiveMix utilizes mixed input-output pairs from the last iteration to generate new synthetic examples, increasing data diversity and encouraging the model to learn from multi-scale and multi-space views. Furthermore, a consistency loss for matching spatial semantics between newly generated samples and the last historical inputs is designed to learn scale-invariant and space-invariant representation.

 
 
 In general, training with plausibility data can improve performance on in-distribution data while increasing the diversity of the created data, which can enhance the model’s robustness to OOD data and adversarial attacks.

 
 
 

#### 3.2.4. Border Smooth

 
 Another problem of Cutmix is the "strong-edge" issue, which refers to the sudden pixel change around the bounding box (i.e., the rectangle’s border). For example, part of the object in the source images could be cut and then pasted on the region of the target image where the object is located. This issue can result in a noticeable boundary or inconsistency in the augmented image.

 
 
 To solve this problem, the construction of mask 𝐌 \mathbf{M} in SmoothMix  ( Lee et al., 2020b ) relies on some properties such as width, height, and spread of the input feature map. More specifically, 𝐌 \mathbf{M} is defined as the combination of its center point coordinates [ m , n ] [m,n] and spread ζ \zeta in the image space, where m m and n n are randomly sampled from two uniform distributions with the range of width W W and height H H . ζ \zeta denotes the area of the cut patch, and the smoothing region is also from a uniform distribution. Take the circle mask as an example:

 

 
 (14) | 
 | 
 𝐌 w = e − ( r − m ) 2 2 ​ ζ 2 , 𝐌 h = e − ( s − n ) 2 2 ​ ζ 2 , 𝐌 = 𝐌 w ⊗ 𝐌 h , \mathbf{M}_{w}=e^{-\frac{(r-m)^{2}}{2\zeta^{2}}},\ \mathbf{M}_{h}=e^{-\frac{(s-n)^{2}}{2\zeta^{2}}},\ \mathbf{M}=\mathbf{M}_{w}\otimes\mathbf{M}_{h}, | 
 | 
 

 where ⊗ \otimes is the outer product and r / s r/s is the pixel in the width / height dimension. In this way, two inputs are mixed with a smooth transition due to the border gradually diminishing outward. Park et al. (2022c) present a Hybrid strategy of the Mixup and Cutmix (HMix) and a Gaussian Mixup (GMix) scheme. HMix cuts and pastes two samples as Cutmix and then linearly interpolates two images in the area out of the cropped box of Cutmix. In contrast, GMix randomly selects a point and then blends two instances gradually. It uses the Gaussian distribution to relax the Cutmix box condition to a continuous transition, preventing the rectangle cropping of Cutmix from generating implausible synthetic data.

 
 
 

#### 3.2.5. Other Cutmix Techniques

 
 Apart from the approaches mentioned above, there are other methods to deal with different problems in Cutmix. Similar to Manifold Mixup  ( Verma et al., 2019a ) , PatchUp  ( Faramarzi et al., 2022 ) carries out interpolation in feature space – selecting adjacent regions of feature maps for mixing two examples. The benefit of Cutmix-based techniques on transformer -based methods  ( Vaswani et al., 2017 ) is limited since the transformer inherently has a global receptive field, which forces the model to learn from a global perspective. To cope with this issue, Liu et al. (2022c) put forward TokenMix in which the mix of two images is token-level by dividing the input into multiple smaller sub-regions and assigning labels of combined images with CAM from a pre-trained teacher model. A similar work is ScoreMix  ( Stegmüller et al., 2023 ) , where the semantic information of images is based on learned self-attention. GridMix  ( Baek et al., 2021 ) splits an image into q × q q\times q grids and then constructs instances by randomly selecting a local cell from one of the two inputs. The mixed label is determined by blending the label pair in proportion to the number of grids. Besides, an additional task predicting the label of each cell is integrated to extract discriminative representations. One step further, PatchMix  ( Cascante-Bonilla et al., 2021 ) presents a grid-level mix data augmentation method with a genetic search scheme to find the optimal patch mask. To exploit the spatial context information in remote sensing semantic segmentation data, ChessMix  ( Pereira and dos Santos, 2021 ) combines transformed mini-patches in a chessboard-like grid to synthesize examples and assigns patches with more examples of the rarest classes the high weight to deal with the imbalance issue. For unbalanced classification, Intra-Class Cutmix  ( Zhao and Lei, 2021 ) boosts model performance by blending intra-class samples of minority classes to modify the decision boundary.

 
 
 Table 4. Summary of Cutmix-based methods 
 
 
 Adaptation | 
 Method | 
 Characteristic | 

 
 Integration with Saliency Information | 
 Attentive Cutmix  ( Walawalkar et al., 2020 ) | 
 Select illustrative regions using feature attention map | 

 
 FocusMix  ( Kim et al., 2020c ) | 
 Sample informative pixels | 

 
 TransMix  ( Chen et al., 2022b ) | 
 Mix targets using attention maps from ViTs | 

 
 SaliencyMix  ( Uddin et al., 2021 ) | 
 Cut indicative regions using saliency detector | 

 
 Puzzle Mix  ( Kim et al., 2020a ) | 
 Find optimal mask and transport | 

 
 SSMix  ( Yoon et al., 2021a ) | 
 Maintain input locality via span-based combination | 

 
 SnapMix  ( Huang et al., 2021 ) | 
 Compose images using CAM to reduce label noise | 

 
 Attribute Mix  ( Li et al., 2020c ) | 
 Use attribute-level semantic information | 

 
 ResizeMix  ( Qin et al., 2020 ) | 
 Shrink one input and paste on the other | 

 
 Improved Divergence | 
 Saliency Grafting  ( Park et al., 2022b ) | 
 Scale saliency map and sample regions using Bernoulli | 

 
 Co-Mixup  ( Kim et al., 2021 ) | 
 Maximize saliency and supermodular diversity | 

 
 RecursiveMix  ( Yang et al., 2022b ) | 
 Use historical Input-Prediction-Output triplets | 

 
 Border Smooth | 
 SmoothMix  ( Lee et al., 2020b ) | 
 Construct mask based on image properties | 

 
 HMix  ( Park et al., 2022c ) | 
 Cutmix and then Mixup outside cropped box | 

 
 GMix  ( Park et al., 2022c ) | 
 Use Gaussian to relax Cutmix to continuous transition | 

 
 Others | 
 PatchUp  ( Faramarzi et al., 2022 ) | 
 Mix intermediate hidden representations | 

 
 TokenMix  ( Liu et al., 2022c ) | 
 Mix images at token-level | 

 
 ScoreMix  ( Stegmüller et al., 2023 ) | 
 Mix images using learned self-attention semantics | 

 
 GridMix  ( Baek et al., 2021 ) | 
 Split image into grids and Cutmix cells | 

 
 PatchMix  ( Cascante-Bonilla et al., 2021 ) | 
 Genetic search for optimal patch mask | 

 
 ChessMix  ( Pereira and dos Santos, 2021 ) | 
 Combine mini-patches in chessboard-like grid | 

 
 Intra-Class Cutmix | 
 Mix intra-class minority samples | 

 
 
 
 

### 3.3. Beyond Mixup Cutmix

 
 Apart from Mixup-based and Cutmix-based methods, there are other techniques based on the mixing principle. These include mixing with itself, incorporating multiple MixDA approaches, and integrating with other DA methods.

 
 
 Mixing with itself. DJMix  ( Hataya and Nakayama, 2022 ) combines each training example with its discretized one instead of blending two samples, i.e., mixing only one image. Discretization is conducted by the Vector-Quantized Variational AutoEncoder with encoder 𝒢 ⁡ ( ⋅ ) \mathcal{G}(\cdot) in an unsupervised manner:

 

 
 (15) | 
 | 
 𝐱 ~ = λ ​ 𝐱 + ( 1 − λ ) ​ 𝒢 ​ ( 𝐱 ) . \tilde{\mathbf{x}}=\lambda\mathbf{x}+(1-\lambda)\mathcal{G}(\mathbf{x}). | 
 | 
 

 A Jensen–Shannon (JS) divergence-based loss is introduced to map 𝐱 \mathbf{x} and 𝐱 ~ \tilde{\mathbf{x}} closely and obtain consistent representations. CutBlur  ( Yoo et al., 2020 ) mixes two resolution versions of an image by cutting and pasting a high-resolution patch to the corresponding low-resolution image area and vice versa. It enforces the model to learn "how" and "where" to super-resolve an image and understand the degree of resolution.

 
 
 Incorporating multiple MixDA approaches. RandomMix  ( Liu et al., 2022e ) conducts random permutation for data batch to generate two times data and combine multiple mix policies: Mixup  ( Zhang et al., 2018 ) , Cutmix  ( Yun et al., 2019 ) , Fmix  ( Harris et al., 2020 ) , and ResizeMix  ( Qin et al., 2020 ) . Besides, AugRmixAT  ( Liu et al., 2022d ) generated multiple different sets of augmented data for a single example by combining multiple MixDA  ( Zhang et al., 2018 ; Yun et al., 2019 ; Harris et al., 2020 ; Qin et al., 2020 ) and adversarial perturbation  ( Szegedy et al., 2014 ) to boost generalization and robustness.

 
 
 Integrating with other DA methods. Through the lens of irregular superpixel decomposition, Hammoudi et al. propose SuperpixelGridMix  ( Hammoudi et al., 2022 ) , a new style of data augmentation method that can be combined with mix-based approaches to form their variants. AugMix  ( Hendrycks et al., 2020 ) presents a data augmentation pipeline with operations from AutoAugment  ( Cubuk et al., 2019 ) . AugMix consists of 3 3 sub-chains, each of which is a randomly selected conventional SsDA maneuver. Mixup then combines the generated images with a mix ratio drawn from a Dirichlet distribution Dirichlet ⁡ ( α , α , α ) \operatorname{Dirichlet}(\alpha,\alpha,\alpha) . "skip connection" with Mixup is used to blend the images from sub-chains and the original image. Besides, a JS divergence-based loss function is designed to provide a consistent representation. StackMix  ( Chen et al., 2022a ) is an orthogonal work that takes input as the concatenation of two images and the target as the average of two one-hot vectors. It can be combined with mix-based methods. For example, Mixup with StackMix can generate new data based on four inputs, enlarging the space of augmented data compared to the original Mixup that cannot benefit from blending more than two inputs  ( Zhang et al., 2018 ) .

 
 
 

### 3.4. Discussion

 
 The primary distinction between Mixup and Cutmix lies in their methodologies for combining samples: Mixup integrates samples globally, whereas Cutmix does so locally. Due to its global blending strategy, Mixup is more suited for tasks demanding a comprehensive grasp of the entire image, such as image classification. In contrast, Cutmix, with its localized approach, excels in tasks like object localization, where detailed attention to specific image regions is crucial. Moreover, when the patch used in Cutmix is not informative relative to the source image, it may fail to enhance performance and introduce noise, destabilizing the training process by creating a biased target. Empirical evidence from experiments using ResNet-50 on CIFAR-100 shows that Mixup achieves a top-1 accuracy of 82.10 % 82.10\% , slightly outperforming Cutmix’s 81.67 % 81.67\% . On the CUB200-2011 benchmarks, however, Cutmix surpasses Mixup in localization accuracy, achieving 54.81 % 54.81\% compared to Mixup’s 49.30 % 49.30\% . This is likely because the global approach of Mixup leads to more significant ambiguity in sample generation compared to Cutmix, prompting models trained with Mixup to concentrate on smaller, more discriminating regions of objects. While beneficial for classification, this focus reduces performance in tasks requiring precise localization.

 
 
 In our detailed analysis of Mixup and Cutmix adaptations, we assess their performance enhancements by integrating them with ResNet-50 and testing on the CIFAR-100 benchmark, as summarized in Table  5 . Notably, each adaptation offers distinct advantages: the "Adaptive Mix Strategy" most significantly improves Mixup by addressing its inherent challenge of a random mixing ratio. Additionally, enhancements like "Diversity" and "Mixing in Embedding Space" are particularly effective, as they aim to augment data variability and leverage more discriminative embeddings, respectively. Conversely, Cutmix benefits more from adaptations such as "Integration with Saliency Information" and "Border Smooth," which mitigate the abrupt transitions inherent to its patch-based approach. However, it is essential to recognize that each modification introduces its challenges. For instance, "Mixing in Embedding Space" tends to slow convergence due to the complexity of selecting appropriate embedding layers for mixing. Similarly, increasing "Diversity in Mixup" and "Improved Divergence" can cause inconsistencies across different data batches. Moreover, adaptations like "Saliency Style Guidance" and "Integration with Saliency Information" necessitate meticulous fine-tuning of hyper-parameters, adding complexity to the data augmentation process.

 
 
 Table 5. Comparisons within Mixup-based and Cutmix-based data augmentation methods. All results are the top-1 accuracies on the CIFAR-100  ( Krizhevsky et al., 2009 ) benchmark with ResNet-50  ( He et al., 2016 ) . 
 
 
 Category | 
 Adaptation | 
 Methods | 
 Performance | 

 
 Mixup | 
 Mixing in Embedding Space | 
 Manifold Mixup  ( Verma et al., 2019a ) | 
 83.23 % 83.23\% | 

 
 NFM  ( Lim et al., 2022 ) | 
 83.74 % 83.74\% | 

 
 Adaptive Mix Strategy | 
 AdaMixUp  ( Guo et al., 2019b ) | 
 84.48 % 84.48\% | 

 
 AutoMix  ( Liu et al., 2022b ) | 
 84.35 % 84.35\% | 

 
 Sample Selection | 
 DMix  ( Li et al., 2021b ) | 
 83.07 % 83.07\% | 

 
 GenLabel  ( Sohn et al., 2022 ) | 
 83.35 % 83.35\% | 

 
 Saliency Style Guidance | 
 StyleMix  ( Hong et al., 2021 ) | 
 82.34 % 82.34\% | 

 
 Mixstyle  ( Zhou et al., 2021 ) | 
 82.69 % 82.69\% | 

 
 Diversity | 
 MixMo  ( Ramé et al., 2021 ) | 
 83.67 % 83.67\% | 

 
 PixMix  ( Hendrycks et al., 2022 ) | 
 83.64 % 83.64\% | 

 
 Cutmix | 
 Integration with Saliency Information | 
 SaliencyMix  ( Uddin et al., 2021 ) | 
 83.66 % 83.66\% | 

 
 Puzzle Mix  ( Kim et al., 2020a ) | 
 83.42 % 83.42\% | 

 
 Improved Divergence | 
 Co-Mixup  ( Kim et al., 2021 ) | 
 82.47 % 82.47\% | 

 
 RecursiveMix  ( Yang et al., 2022b ) | 
 83.01 % 83.01\% | 

 
 Border Smooth | 
 SmoothMix  ( Lee et al., 2020b ) | 
 83.87 % 83.87\% | 

 
 HMix  ( Park et al., 2022c ) | 
 84.02 % 84.02\% | 

 
 
 
 

## 4. MixDA Applications

 
 In this section, we will review extensive applications of MixDA.

 
 

### 4.1. Semi-Supervised Learning

 
 Semi-supervised learning (SSL) has made significant strides in various domains where unlabeled data is abundant while labeling such data remains challenging due to the need for human resources and expertise.

 
 
 To integrate MixDA into SSL, Verma et al. (2019b) propose Interpolation Consistency Training (ICT) by transiting the decision boundary to low-density regions of data distribution. Specifically, ICT forces the prediction of the interpolation of unlabeled samples to be consistent with the interpolation of the model predictions:

 

 
 (16) | 
 | 
 f ⁡ ( λ ​ 𝐮 i + ( 1 − λ ) ​ 𝐮 j ) = λ ​ f ​ ( 𝐮 i ) + ( 1 − λ ) ​ f ​ ( 𝐮 j ) , f(\lambda\mathbf{u}_{i}+(1-\lambda)\mathbf{u}_{j})=\lambda f(\mathbf{u}_{i})+(1-\lambda)f(\mathbf{u}_{j}), | 
 | 
 

 where 𝐮 i \mathbf{u}_{i} and 𝐮 j \mathbf{u}_{j} are sampled unlabeled examples, and f ⁡ ( 𝐮 i ) f(\mathbf{u}_{i}) and f ⁡ ( 𝐮 j ) f(\mathbf{u}_{j}) are the corresponding model predictions. ICT regards the prediction of an unlabeled instance as its target (i.e., guessed label). Similarly, Olsson et al. (2021) present ClassMix, which combines unlabeled data to construct new examples. The mixed image is generated by cutting half of one image and pasting it on the other. Its corresponding target is computed based on the network’s semantic prediction.

 
 
 Another independent work is MixMatch  ( Berthelot et al., 2019b ) , which generates synthetic examples using a batch of labeled samples and another batch of unlabeled instances with guessed labels. Different from ICT, which can be regarded as a case of MixMatch where only unlabeled data is mixed, Mixup is utilized to generate mixed labeled samples and mixed unlabeled examples in MixMatch. One step further, ReMixMatch  ( Berthelot et al., 2019a ) improves the semi-supervised learning performance of MixMatch by distribution alignment and augmentation anchoring. To avoid confirmation bias (i.e., error accumulation) of self-training in MixMatch, DivideMix  ( Li et al., 2020a ) rejects the sample’s label with high noise and regards it as unlabeled data to improve generalization. Specifically, DivideMix trains two individual Gaussian mixture models (GMMs) to separate the training data into a label set and an unlabeled collection, which will be used to train the main network.

 
 
 

### 4.2. Contrastive Learning

 
 Contrastive Learning (CL) has emerged as a prominent approach within self-supervised learning, focusing on the principle of learning discriminative features by contrasting positive and negative samples. According to Liu et al. (2021b) , the core idea is to bring closer the representations of augmented versions (positive examples) of the same data point (anchor sample) and simultaneously drive apart the embeddings of different data points (negative samples). CL’s operational mechanism can be illustrated by formulating the InfoNCE contrastive loss. For a given anchor sample 𝐱 i \mathbf{x}_{i} , its augmented version 𝐱 i ′ \mathbf{x}^{\prime}_{i} serves as the positive sample, the set of embeddings from negative samples, denoted as 𝒬 i \mathcal{Q}_{i} , the InfoNCE loss ℒ InfoNCE \mathcal{L}_{\text{InfoNCE}} is mathematically defined as:

 

 
 (17) | 
 | 
 𝐡 i = 𝒢 ( 𝐱 i ) , 𝐡 ′ i = 𝒢 ( 𝐱 ′ i ) , 𝐡 k = 𝒢 ⁡ ( 𝐱 k ) ​ ( 𝐱 k ∈ 𝒬 i ) ℒ InfoNCE = − log ⁡ e ( 𝐡 i ​ 𝐡 i ′ / τ ) ∑ 𝐡 k e ( 𝐡 i ​ 𝐡 k / τ ) , \begin{gathered}\mathbf{h}_{i}=\mathcal{G}(\mathbf{x}_{i}),\ \mathbf{h}^{\prime}_{i}=\mathcal{G}(\mathbf{x}^{\prime}_{i}),\\
\ \mathbf{h}_{k}=\mathcal{G}(\mathbf{x}_{k})\ (\mathbf{x}_{k}\in\mathcal{Q}_{i})\\
\mathcal{L}_{\text{InfoNCE}}=-\log\frac{e^{(\mathbf{h}_{i}\mathbf{h}^{\prime}_{i}/\tau)}}{\sum_{\mathbf{h}_{k}}e^{(\mathbf{h}_{i}\mathbf{h}_{k}/\tau)}},\end{gathered} | 
 | 
 

 where τ \tau is the temperature factor. The minimization of the InfoNCE loss is central to learning useful representations. By maximizing the similarity between an anchor and its positive sample while minimizing the similarity between the anchor and the negative samples, the model learns to embed similar items closer together in the feature space, even without explicit labels. A straightforward MixDA-based method is Mixup Contrast (MixCo)  ( Kim et al., 2020b ) , which introduces  semi-positive samples that are decoded from the mixture of positive and negative instances to learn fine-grained similarity between representations. The semi-positive sample 𝐱 ~ \tilde{\mathbf{x}} is generated by MixDA:

 

 
 (18) | 
 | 
 𝐱 ~ = λ ​ 𝐱 i + ( 1 − λ ) ​ 𝐱 k , 𝐡 ~ = 𝒢 ⁡ ( 𝐱 ~ ) , \tilde{\mathbf{x}}=\lambda\mathbf{x}_{i}+(1-\lambda)\mathbf{x}_{k},\ \tilde{\mathbf{h}}=\mathcal{G}(\tilde{\mathbf{x}}), | 
 | 
 

 where 𝐱 k \mathbf{x}_{k} is from the negative sample set 𝒬 i \mathcal{Q}_{i} . The mix 𝐱 ~ \tilde{\mathbf{x}} is the semi-positive sample for both 𝐱 i \mathbf{x}_{i} and 𝐱 k \mathbf{x}_{k} , with similarity as λ \lambda and ( 1 − λ ) (1-\lambda) , respectively. Therefore, another loss function is derived:

 

 
 (19) | 
 | 
 ℒ MixCo = − [ λ ​ log ⁡ e ( 𝐡 ~ ​ 𝐡 i ′ / τ ) ∑ 𝐡 k e ( 𝐡 ~ ​ 𝐡 k / τ ) + ( 1 − λ ) ​ log ⁡ e ( 𝐡 ~ ​ 𝐡 i ′ / τ ) ∑ 𝐡 k e ( 𝐡 ~ ​ 𝐡 k / τ ) ] . \mathcal{L}_{\text{MixCo}}=-[\lambda\log\frac{e^{(\tilde{\mathbf{h}}\mathbf{h}^{\prime}_{i}/\tau)}}{\sum_{\mathbf{h}_{k}}e^{(\tilde{\mathbf{h}}\mathbf{h}_{k}/\tau)}}+(1-\lambda)\log\frac{e^{(\tilde{\mathbf{h}}\mathbf{h}^{\prime}_{i}/\tau)}}{\sum_{\mathbf{h}_{k}}e^{(\tilde{\mathbf{h}}\mathbf{h}_{k}/\tau)}}]. | 
 | 
 

 By combining loss ℒ InfoNCE \mathcal{L}_{\text{InfoNCE}} and loss ℒ MixCo \mathcal{L}_{\text{MixCo}} , the model can capture such a semi-positive relation, which is much more complicated than merely discriminating positives from the negatives, leading to more efficient use of negatives. Zhang et al. (2021b) demonstrate that learning with contrastive loss benefits representation learning and fine-tuning. Contrast-Regularized tuning (Core-tuning) is proposed to fine-tune contrastive self-supervised learning models. A hard pair excavation scheme is applied to learn transferable features and smooth the decision boundary. In particular, Mixup is used to generate (i) hard positive pair by mixing the two hardest pairs and (ii) semi-hard negative pair by blending a negative example and the anchor data. Other similar works include Feature Transformation (FT)  ( Zhu et al., 2021 ) , mixing of contrastive hard negatives (MoCHi)  ( Kalantidis et al., 2020 ) , and background mixing  ( Sahoo et al., 2021b ) .

 
 
 Following the Single Instance Multi-view (SIM) paradigm, Siamese network  ( Chen and He, 2021 ) takes the augmented images of an example as inputs and conducts an element-wise maximum of features to compute the discriminative representation. The loss function for SIM ℒ S ​ I ​ M \mathcal{L}_{SIM} is as follows:

 

 
 (20) | 
 | 
 𝐡 i = 𝒢 ( 𝐱 i ) , 𝐡 ′ i = 𝒢 ( 𝐱 ′ i ) , 𝐳 i = 𝒫 ( 𝐡 i ) , 𝐳 ′ i = 𝒫 ( 𝐡 ′ i ) , 𝒟 ( 𝐡 i , 𝐳 ′ i ) = − 𝐡 i ‖ 𝐡 i ‖ 2 ⋅ 𝐳 i ′ ‖ 𝐳 i ′ ‖ 2 , 𝒟 ( 𝐡 ′ i , 𝐳 i ) = − 𝐡 i ′ ‖ 𝐡 i ′ ‖ 2 ⋅ 𝐳 i ‖ 𝐳 i ‖ 2 , ℒ SIM = 1 2 ​ 𝒟 ​ ( 𝐡 i , 𝐳 i ′ ) + 1 2 ​ 𝒟 ​ ( 𝐡 i ′ , 𝐳 i ) , \begin{gathered}\mathbf{h}_{i}=\mathcal{G}(\mathbf{x}_{i}),\ \mathbf{h}^{\prime}_{i}=\mathcal{G}(\mathbf{x}^{\prime}_{i}),\\
\mathbf{z}_{i}=\mathcal{P}(\mathbf{h}_{i}),\ \mathbf{z}^{\prime}_{i}=\mathcal{P}(\mathbf{h}^{\prime}_{i}),\\
\mathcal{D}(\mathbf{h}_{i},\mathbf{z}^{\prime}_{i})=-\frac{\mathbf{h}_{i}}{\|\mathbf{h}_{i}\|_{2}}\cdot\frac{\mathbf{z}^{\prime}_{i}}{\|\mathbf{z}^{\prime}_{i}\|_{2}},\ \mathcal{D}(\mathbf{h}^{\prime}_{i},\mathbf{z}_{i})=-\frac{\mathbf{h}^{\prime}_{i}}{\|\mathbf{h}^{\prime}_{i}\|_{2}}\cdot\frac{\mathbf{z}_{i}}{\|\mathbf{z}_{i}\|_{2}},\\
\mathcal{L}_{\text{SIM}}=\frac{1}{2}\mathcal{D}(\mathbf{h}_{i},\mathbf{z}^{\prime}_{i})+\frac{1}{2}\mathcal{D}(\mathbf{h}^{\prime}_{i},\mathbf{z}_{i}),\end{gathered} | 
 | 
 

 where 𝒫 \mathcal{P} is the prediction head. Guo et al.  ( Guo et al., 2021 ) present MixSiam, a MixDA that feeds the combinations of augmented images into the backbone and encourages the embeddings to be close to the original distinctive representation:

 

 
 (21) | 
 | 
 𝐱 ~ = λ ​ 𝐱 i + ( 1 − λ ) ​ 𝐱 i ′ , 𝐡 i = 𝒢 ( 𝐱 i ) , 𝐡 ′ i = 𝒢 ( 𝐱 ′ i ) , 𝐡 ~ = 𝒢 ( 𝐱 ~ ) , 𝐳 i = 𝒫 ( 𝐡 i ) , 𝐳 ′ i = 𝒫 ( 𝐡 ′ i ) , 𝐳 ~ = Max ( 𝐳 i , 𝐳 ′ i ) , ℒ MixSiam = 𝒟 ( 𝐡 ~ , 𝐳 ~ ) = − 𝐡 ~ ‖ 𝐡 ~ ‖ 2 ⋅ 𝐳 ~ ‖ 𝐳 ~ ‖ 2 , \begin{gathered}\tilde{\mathbf{x}}=\lambda\mathbf{x}_{i}+(1-\lambda)\mathbf{x}^{\prime}_{i},\\
\mathbf{h}_{i}=\mathcal{G}(\mathbf{x}_{i}),\ \mathbf{h}^{\prime}_{i}=\mathcal{G}(\mathbf{x}^{\prime}_{i}),\ \tilde{\mathbf{h}}=\mathcal{G}(\tilde{\mathbf{x}}),\\
\mathbf{z}_{i}=\mathcal{P}(\mathbf{h}_{i}),\ \mathbf{z}^{\prime}_{i}=\mathcal{P}(\mathbf{h}^{\prime}_{i}),\ \tilde{\mathbf{z}}=\operatorname{Max}(\mathbf{z}_{i},\mathbf{z}^{\prime}_{i}),\\
\mathcal{L}_{\text{MixSiam}}=\mathcal{D}(\tilde{\mathbf{h}},\tilde{\mathbf{z}})=-\frac{\tilde{\mathbf{h}}}{\|\tilde{\mathbf{h}}\|_{2}}\cdot\frac{\tilde{\mathbf{z}}}{\|\tilde{\mathbf{z}}\|_{2}},\end{gathered} | 
 | 
 

 where Max \operatorname{Max} denotes the element-wise maximum. By combining ℒ MixSiam \mathcal{L}_{\text{MixSiam}} and ℒ SIM \mathcal{L}_{\text{SIM}} , MixSiam increases the diversity of the augmented sample and keeps predicted embeddings invariant. However, Siamese frameworks are prone to overfitting if the augmentation technique used to construct two views is not expressive enough. To deal with this problem, unsupervised image mixtures (Un-Mix)  ( Shen et al., 2022 ) entails the soft distance concept into the label space and makes the model embrace the soft degree of similarity between image pairs by interpolating in the input space. Specifically, Un-Mix first generates two views with a pre-defined transformation, followed by a mix operation. Finally, the mixed samples are fed into the two-branch framework to produce hidden representations.

 
 
 Going beyond SIM learning, BSIM  ( Chu et al., 2020 ) takes the similarity of spurious-positive pairs (i.e., two randomly sampled instances and their mixture) into account for more even feature distribution, enhancing the discrimination capability of the model. Contrastive learning is recast as learning a non-parametric classifier that assigns a virtual class to each sample in work  ( Lee et al., 2021b ) , which uses i i -Mix to give a virtual label to each sample and conducts interpolation in both input space and virtual label space to regularize the contrastive representation learning. MCL  ( Wickstrøm et al., 2022 ) designs a novel contrastive loss function to take advantage of the MixDA strategy, which predicts the mixing coefficient regarded as the soft target in the loss function. Li et al. (2020b) present CLIM, which finds local examples similar to a cluster center and utilizes a mix scheme to perform smoothing regularization to deduce powerful representation. Simple data mixing prior (SDMP)  ( Ren et al., 2022 ) considers the connections between source images and the generated counterparts and encodes the relationships into hidden space to model this prior. Besides, Similarity Mixup  ( Patel et al., 2022 ) virtually increases the batch size and operates on pair-wise scalar similarities. Many constructed hard negatives in Graph Contrastive Learning (GCL) are positive samples, which will undermine the performance. To resolve this problem, ProGCL-Mix  ( Xia et al., 2022 ) is proposed to generate more hard negatives considering the probability of a negative being the true one.

 
 
 Overall, MixDA is utilized to enhance Contrastive Learning (CL) in two main ways: (i) generating diverse mixed examples, such as semi-hard negative pairs and semi-positive samples, to incorporate finer-grained information and facilitate the learning of more discriminative representations and (ii) integrating MixDA into the optimization process as prior knowledge to regularize the CL objective.

 
 
 

### 4.3. Adversarial Training

 
 Adversarial training  ( Szegedy et al., 2014 ) can significantly lift model robustness since it encourages the model to explore some unseen regions. This is achieved by perturbing a training data point in the direction of another instance while keeping its origin target as the training label, improving generalization.

 
 
 Through the lens of optimization, Madry et al. (2018) formulate adversarial robustness as a min-max problem:

 

 
 (22) | 
 | 
 f ∗ = arg ⁡ min ⁡ ∫ f ∈ ℱ ⁡ max δ ∈ 𝒮 ⁡ ℒ ⁡ ( f ⁡ ( 𝐱 + δ ) , 𝐲 ) ​ d ​ P ψ ​ ( 𝐱 , 𝐲 ) , f^{*}=\arg\min_{f\in\mathcal{F}}\int\max_{\delta\in\mathcal{S}}\mathcal{L}(f(\mathbf{x}+\delta),\mathbf{y})\mathrm{d}P_{\psi}(\mathbf{x},\mathbf{y}), | 
 | 
 

 where 𝒮 \mathcal{S} is the neighborhood region of each sample. P ψ ​ ( 𝐱 , 𝐲 ) = 1 N ​ ∑ i = 1 N ρ ⁡ ( 𝐱 = 𝐱 i , 𝐲 = 𝐲 i ) P_{\psi}(\mathbf{x},\mathbf{y})=\frac{1}{N}\sum_{i=1}^{N}\rho(\mathbf{x}=\mathbf{x}_{i},\mathbf{y}=\mathbf{y}_{i}) in which ρ ⁡ ( 𝐱 = 𝐱 i , 𝐲 = 𝐲 i ) \rho(\mathbf{x}=\mathbf{x}_{i},\mathbf{y}=\mathbf{y}_{i}) is a Dirac mass centered at ( 𝐱 i , 𝐲 i ) (\mathbf{x}_{i},\mathbf{y}_{i}) . It is deliberately designed to have visually imperceptible perturbations δ \delta only. Adversarial training consists of two procedures: (i) generating adversarial examples and (ii) training the model to assign the origin target to augmented data for improved robustness against unseen adversarial examples. The similarity between adversarial training and MixDA is formulated by reinterpreting MixDA as a kind of Directional Adversarial Training (DAT)  ( Archambault et al., 2019 ) , which perturbs one training sample based on the direction of the other instance as the same expected loss functions they have. Meanwhile, Empirical Risk Minimization (ERM) is notorious as it can be easily fooled by adversarial examples  ( Goodfellow et al., 2015 ; Szegedy et al., 2014 ) . In contrast, MixDA can be regarded as Vicinal Risk Minimization (VRM)  ( Zhang et al., 2018 ) . Besides, from the view of robust objectives and the optimization dynamic for neural networks, adversarial training needs more training data to gain better generalization performance  ( Schmidt et al., 2018 ) . Thus, deeply investigating the connection between MixDA and adversarial training would be natural.

 
 
 By investigating the mutual information between the function learned on the original data and the counterpart on the mixed data by a VAE, Harris et al.  ( Harris et al., 2020 ) demonstrate that MixDA approaches can be regarded as a form of adversarial training  ( Goodfellow et al., 2015 ) , therefore, enhancing robustness to attacks such as injecting uniform noise to create examples similar to those synthesized by Mixup. Zhang et al. (2021a) attribute the adversarial robustness gain of mix training to the approximate loss function, which acts as an upper bound of the adversarial loss with the ℓ 2 \ell_{2} attack. For robustness, minimizing the loss function with augmented instances is approximately equivalent to minimizing an upper bound of the adversarial loss. This explains why models trained with mixed data manifest robustness improvement for several adversarial attacks. To increase training data, Interpolated Adversarial Training (IAT)  ( Lamb et al., 2019 ) trains the model on the mixture of adversarial samples along with the combination of unperturbed instances. Similarly, a soft-labeled approach for improved adversarial generalization, Adversarial Vertex Mixup (AVMixup)  ( Lee et al., 2020a ) , is presented to extend the training distribution via the interpolation between the raw input vector and the virtual vector (adversarial vertex) defined in the adversarial direction. Si et al. (2021) put forward Adversarial and Mixup Data Augmentation (AMDA) to explore a much larger attack search space by linearly interpolating the representation pair to generate augmented data. The generated data are more diverse and abundant than discrete text adversarial examples generated by general adversarial methods. Taking advantage of Mixup, M-TLAT  ( Laugros et al., 2020 ) designs a novel adversarial training algorithm called Targeted Labeling Adversarial Training (TLAT) to boost the robustness of image classifiers for 19 19 common corruptions and 5 5 adversarial attacks, without sacrificing the accuracy on clean samples. Specifically, M-TLAT interpolates the target labels of adversarial samples with the ground-truth labels:

 

 
 (23) | 
 | 
 𝐱 mtlat = 𝐱 ~ − arg ⁡ max δ ∈ 𝒮 ​ { ℒ ⁡ ( 𝐱 ~ + δ , 𝐲 target ) } , 𝐲 mtlat = ( 1 − δ ) ∗ 𝐲 ~ + δ ∗ 𝐲 target , \mathbf{x}_{\text{mtlat}}=\tilde{\mathbf{x}}-\arg\max_{\delta\in\mathcal{S}}\{\mathcal{L}(\tilde{\mathbf{x}}+\delta,\mathbf{y}_{\text{target}})\},\ \mathbf{y}_{\text{mtlat}}=(1-\delta)*\tilde{\mathbf{y}}+\delta*\mathbf{y}_{\text{target}}, | 
 | 
 

 where ( 𝐱 ~ , 𝐲 ~ ) (\tilde{\mathbf{x}},\tilde{\mathbf{y}}) is the Mixup-ed input-output pair and 𝐲 target \mathbf{y}_{\text{target}} is the targeted label for adversarial attack. 𝐱 mtlat \mathbf{x}_{\text{mtlat}} contains features from three different sources: two clean images specified by 𝐱 ~ \tilde{\mathbf{x}} and an adversarial perturbation targeting a specific class 𝐲 target \mathbf{y}_{\text{target}} . Correspondingly, label 𝐲 mtlat \mathbf{y}_{\text{mtlat}} embraces their supervised signals with the associated weights, which are jointly decided by the mix ratio λ \lambda and the perturbation δ \delta . M-TLAT predicts the label corresponding to the three sources while outputting the weight of each source within the combined feature 𝐱 mtlat \mathbf{x}_{\text{mtlat}} , encouraging the model to learn a subtle representation of mixed images. Bunk et al.  ( Bunk et al., 2021 ) combine Mixup and adversarial training by mining the space between samples using projected gradient descent. It utilizes back-propagation through the Mixup interpolation in the training stage to optimize for the matte and incongruous region. Besides, the effect of Mixup ratio optimization is also investigated to improve robustness. Similarly, MixACM  ( Muhammad et al., 2021 ) aligns activated channel maps (ACM) by transferring the robustness of a teacher to a student. Mixup-SSAT  ( Jiao et al., 2023 ) mixes the adversarial and benign scenarios in an adversarial training manner to gain a trade-off between accuracy and adversarial training. Moreover, an approach called Clean Feature Mixup (CFM) is introduced by  Byun et al. (2023) to enhance the transferability of targeted adversarial examples.

 
 
 The methods mentioned above implicitly defend against adversarial attacks in the testing stage by directly classifying the test data in a vanilla model trained on mixed data. In contrast, an inference principle called Mixup Inference (MI)  ( Pang et al., 2020 ) exploits the induced global linearity in the trained model. The basic idea is that breaking the locality via the globality of the model predictions would benefit the adversarial robustness. MI mixes one input with some other randomly chosen clean samples. They can be transformed into equivalent perturbations, destroying the locality of adversarial attacks and reducing their strength. Specifically, MI first samples a label 𝐲 s \mathbf{y}_{s} and then samples K K examples { 𝐱 s , k } k = 1 K \{\mathbf{x}_{s,k}\}^{K}_{k=1} from the sampled class. In the inference stage, Mixup is used to generate input 𝐱 ~ s , k = λ ​ 𝐱 + ( 1 − λ ) ​ 𝐱 s , k \tilde{\mathbf{x}}_{s,k}=\lambda\mathbf{x}+(1-\lambda)\mathbf{x}_{s,k} and the final result is averaged on the output predictions of the K K Mixup-ed examples:

 

 
 (24) | 
 | 
 f ⁡ ( 𝐱 ) = f ⁡ ( 𝐱 ) + 1 K ​ f ​ ( 𝐱 ~ s , k ) . f(\mathbf{x})=f(\mathbf{x})+\frac{1}{K}f(\tilde{\mathbf{x}}_{s,k}). | 
 | 
 

 
 
 

### 4.4. Generative Models

 
 Generative models, such as GANs  ( Goodfellow et al., 2014 ) , VAE  ( Kingma and Welling, 2014 ) , normalizing flows  ( Kobyzev et al., 2020 ) , and denoising diffusion probabilistic models (DDPM)  ( Ho et al., 2020 ) , learn underlying probability distribution to understand data and generate new instances with new configurations of latent presentation. An autoencoder model ℱ ⁡ ( ⋅ ) \mathcal{F}(\cdot) consists of an encoder 𝒢 ⁡ ( ⋅ ) \mathcal{G}(\cdot) and a decoder ℋ ⁡ ( ⋅ ) \mathcal{H}(\cdot) , i.e., ℱ ⁡ ( ⋅ ) \mathcal{F}(\cdot) = ℋ ⁡ ( 𝒢 ⁡ ( ⋅ ) ) \mathcal{H}(\mathcal{G}(\cdot)) . An input 𝐱 \mathbf{x} is encoded as a hidden embedding 𝐳 \mathbf{z} via 𝒢 \mathcal{G} and then reconstructed by ℋ \mathcal{H} .

 
 
 Applying MixDA in the generative models is straightforward. For example, pseudo input is generated by Mixup and then leveraged to break the evidence lower bound (ELBO) value bottleneck by reducing the margin between ELBO and the data likelihood  ( Feng et al., 2021b ) . Another two direct applications of MixDA are adversarial autoencoder (AAE)  ( Liu et al., 2018 ) and VarMixup  ( Mangla et al., 2020 ) . The former interpolates hidden features to construct a broad of latent representations. The latter linearly interpolates on the unfolded latent manifold where the linearity of training data points is preserved.

 
 
 Berthelot et al. (2019c) suggest that, in some cases, autoencoders can be interpolable, i.e., decoding the convex combination of the representations in the hidden space for two data points, the autoencoder can generate semantically mixed instances. Accordingly, Adversarially Constrained Autoencoder Interpolation (ACAI) utilizes a critic network d d to force interpolated data points to be realistic, and the training objective is to minimize:

 

 
 (25) | 
 | 
 ℒ d = ‖ d ⁡ ( ℋ ⁡ ( λ ​ 𝒢 ​ ( 𝐱 i + ( 1 − λ ) ​ 𝒢 ​ ( 𝐱 j ) ) − λ ) ) ‖ 2 + ‖ d ⁡ ( η ​ 𝐱 + ( 1 − η ) ​ ℱ ​ ( 𝐱 ) ) ‖ 2 , \mathcal{L}_{d}=\|d(\mathcal{H}(\lambda\mathcal{G}(\mathbf{x}_{i}+(1-\lambda)\mathcal{G}(\mathbf{x}_{j}))-\lambda))\|^{2}+\ \|d(\eta\mathbf{x}+(1-\eta)\mathcal{F}(\mathbf{x}))\|^{2}, | 
 | 
 

 where η \eta is a scalar hyperparameter. The first term aims to recover the mix ratio used for interpolation in latent space, while the second term is a regularizer with two utilities: (i) it encourages the critic consistently to output 0 0 for non-interpolated inputs and (ii) it enables the critic to obtain realistic data by interpolating in data space, i.e., 𝐱 \mathbf{x} and ℱ ⁡ ( 𝐱 ) \mathcal{F}(\mathbf{x}) , even when the autoencoder’s reconstructions are poor.

 
 
 To improve the quality of reconstructed output, Larsen et al. (2016) combine it with an adversarial game. In particular, a discriminator 𝒟 \mathcal{D} attempts to discern between real input and reconstructed output. In turn, the autoencoder is expected to generate realistic reconstructions to fool the discriminator, formulating a new baseline – AE_GANs :

 

 
 (26) | 
 | 
 min ℱ ⁡ 𝔼 𝐱 ∼ P ​ ‖ 𝐱 − ℱ ⁡ ( 𝐱 ) ‖ 2 ⏟ reconstruction + η ​ ℒ GAN ​ ( 𝒟 ​ ( ℱ ​ ( 𝐱 ) ) , 1 ) ⏟ fool ​ D ​ with reconstruction , \min_{\mathcal{F}}\mathbb{E}_{\mathbf{x}\sim P}\underbrace{\|\mathbf{x}-\mathcal{F}(\mathbf{x})\|_{2}}_{\text{reconstruction}}+\eta\underbrace{\mathcal{L}_{\text{GAN}}(\mathcal{D}(\mathcal{F}(\mathbf{x})),1)}_{\text{fool}\,D\,\text{with reconstruction}}, | 
 | 
 

 

 
 (27) | 
 | 
 min ℱ ⁡ 𝔼 𝐱 ∼ P ​ ℒ GAN ​ ( 𝒟 ​ ( 𝐱 ) , 1 ) ⏟ label ​ 𝐱 ​ a ​ s ​ real + ℒ GAN ​ ( 𝒟 ​ ( ℱ ​ ( 𝐱 ) ) , 0 ) ⏟ label reconstruction as fake , \min_{\mathcal{F}}\mathbb{E}_{\mathbf{x}\sim P}\underbrace{\mathcal{L}_{\text{GAN}}(\mathcal{D}(\mathbf{x}),1)}_{\text{label}\,\mathbf{x}\,{as\,\text{real}}}+\underbrace{\mathcal{L}_{\text{GAN}}(\mathcal{D}(\mathcal{F}(\mathbf{x})),0)}_{\text{label reconstruction as fake}}, | 
 | 
 

 where ℒ GAN \mathcal{L}_{\text{GAN}} is the binary cross-entropy equivalent to the JS GANs  ( Goodfellow et al., 2014 ) . Based on this, adversarial Mixup resynthesis (AMR)  ( Beckham et al., 2019 ) mixes hidden embeddings that are indistinguishable from real samples after decoding. AMR first encodes a hidden pair 𝐡 i = 𝒢 ⁡ ( 𝐱 i ) \mathbf{h}_{i}=\mathcal{G}(\mathbf{x}_{i}) and 𝐡 j = 𝒢 ⁡ ( 𝐱 j ) \mathbf{h}_{j}=\mathcal{G}(\mathbf{x}_{j}) and then mixes them. The mixed representation is decoded by the decoder ℋ ⁡ ( ⋅ ) \mathcal{H}(\cdot) . AMR minimizes a tailored loss function that aims to fool the discriminator 𝒟 \mathcal{D} with output decoded from the constructed instances. Formally, term ℒ GAN ​ ( 𝒟 ⁡ ( ℋ ⁡ ( Mix ⁡ ( 𝐡 i , 𝐡 j ) ) ) , 1 ) \mathcal{L}_{\text{GAN}}(\mathcal{D}(\mathcal{H}(\operatorname{Mix}(\mathbf{h}_{i},\mathbf{h}_{j}))),1) is added to the loss function of autoencoder: Equation ( 26 ), and term ℒ GAN ​ ( 𝒟 ⁡ ( ℋ ⁡ ( Mix ⁡ ( 𝐡 i , 𝐡 j ) ) ) , 0 ) \mathcal{L}_{\text{GAN}}(\mathcal{D}(\mathcal{H}(\operatorname{Mix}(\mathbf{h}_{i},\mathbf{h}_{j}))),0) is appended to the loss function of GANs: Equation ( 27 ).

 
 
 

### 4.5. Domain Adaptation

 
 Domain adaptation involves transferring a model that is trained on a labeled source domain 𝒳 ​ s \mathcal{X}{s} to an unlabeled target domain 𝒰 ​ t \mathcal{U}{t} . Like semi-supervised learning (SSL) discussed earlier, MixDA is also an effective approach for leveraging unlabeled data in domain adaptation. By incorporating MixDA, the model can benefit from the additional information in the unlabeled target domain, leading to improved performance and adaptation to the target domain.

 
 
 Mao et al.  ( Mao et al., 2019 ) present a novel general algorithm called Virtual Mixup Training (VMT) to impose the linear combination constraint on the region in-between training data for improved regularization. Unlike conventionally mixing samples with labels, VMT constructs new data by combining examples without any supervision signal from the target domain. To do this, VMT synthesizes virtual samples with guessed labels as:

 

 
 (28) | 
 | 
 𝐮 ~ = λ ​ 𝐮 i + ( 1 − λ ) ​ 𝐮 j , 𝐲 ~ = λ ​ f ​ ( 𝐮 i ) + ( 1 − λ ) ​ f ​ ( 𝐮 j ) , ℒ = 𝔼 ⁡ [ D KL ⁡ ( 𝐲 ~ , f ⁡ ( 𝐮 ~ ) ) ] , \tilde{\mathbf{u}}=\lambda\mathbf{u}_{i}+(1-\lambda)\mathbf{u}_{j},\ \tilde{\mathbf{y}}=\lambda f(\mathbf{u}_{i})+(1-\lambda)f(\mathbf{u}_{j}),\ \mathcal{L}=\mathbb{E}[\operatorname{D}_{\text{KL}}(\tilde{\mathbf{y}},f(\tilde{\mathbf{u}}))], | 
 | 
 

 where 𝐮 i \mathbf{u}_{i} and 𝐮 j \mathbf{u}_{j} are the samples without labels from the target domain 𝒰 t \mathcal{U}_{t} and D KL \operatorname{D}_{\text{KL}} denotes the KL divergence. However, training VMT with conditional entropy loss may be unstable and may even result in a degenerated solution for some complicated tasks. To tackle this problem, VMT mixes on the input of the softmax layer (i.e., logit values) rather than the output of the softmax layer (i.e., probabilities), as most of the probability values are encouraged to be zero due to the one-hot target vector, while the logits still have non-zero values. Inter- and Intra-domain Mixup training (IIMT)  ( Yan et al., 2020 ) imposes training constraints across domains using Mixup to enhance the generalization performance on the target domain, where adversarial domain learning and Intra-domain Mixup are explored to improve performance further. Another work combining adversarial domain adaptation and Mixup is DM-ADA  ( Xu et al., 2020 ) , which keeps domain invariance in a continuous latent space and trains a domain discriminator to discern examples relative to the source and target domains. The mixed domain label is then used to encourage the domain discriminator to align representations and constrain distance between mixed examples. Besides, domain Mixup is jointly implemented on both pixel and feature levels to lift the model’s robustness. Similarly, Dual Mixup regularized learning (DMRL)  ( Wu et al., 2020 ) forces the model to output coherent predictions to boost the intrinsic structures of the hidden space by inherent structures carrying domain and category Mixup regularization. To mitigate intra-domain discrepancy and inter-domain representation gap, a progressive data interpolation strategy is proposed by  Ma et al. (2023) , incorporating progressive anchor selection and dynamic interpolation rate. The negative transfer – the dearth of domain invariance and distinction of the latent representation – is the main issue in partial domain adaptation, where the label space of the target domain is a subset of the source label space. To deal with this problem, a Select, Label, and Mix (SLM) framework  ( Sahoo et al., 2021a ) is developed to discriminate the invariant feature embedding. The Mix conducts domain Mixup with the Select and Label parts to scrutinize more intrinsic structures across domains, resulting in a domain-invariant hidden space.

 
 
 

### 4.6. Sentence Classification

 
 Conventional DA techniques for text data  ( Feng et al., 2021a ; Li et al., 2022a ) differ from those for image data  ( Yang et al., 2022c ; Shorten and Khoshgoftaar, 2019 ) , which can directly leverage human knowledge for label-invariant data transformation, such as cropping, flipping, and changing the intensity of RGB channels  ( Krizhevsky et al., 2012 ) . For example, slightly altering a word (token) in a sentence may dramatically change its meaning. Therefore, commonly used DA methods for text data are based on synonym replacement from handcrafted ontology word similarity  ( Kobayashi, 2018 ) , inevitably limiting their scope of application because only a small portion of words have precisely or nearly the same meanings. However, MixDA is general enough to generate augmented textual data.

 
 
 As the first MixDA method for text data, Guo et al.  ( Guo et al., 2019a ) adapt Mixup for NLP on the sentence classification task. They propose wordMixup to interpolate word embeddings and senMixup to interpolate sentence embeddings. Experimental evaluations with several network architectures verify their effectiveness. Another similar work is Mixup-Transformer  ( Sun et al., 2020 ) , which incorporates Mixup with Transformer  ( Vaswani et al., 2017 ) architecture. An issue for pre-trained language models is the miscalibration of in-distribution and OOD data due to over-parameterization. A regularized fine-tuning approach is developed for improved calibration  ( Kong et al., 2020 ) . The synthetic on-manifold examples are produced by interpolating in the data manifold to impose a smoothness constraint for better in-distribution calibration. Besides, the predictions for off-manifold instances are forced to be uniformly distributed to mitigate the over-confidence issue for OOD data. Similar to Manifold Mixup  ( Verma et al., 2019a ) , TMix   ( Chen et al., 2020c ) takes in two text instances and interpolates them in their corresponding hidden space. Emix  ( Jindal et al., 2020 ) generates virtual examples by interpolating hidden layer representations with word embeddings and considers the difference in text energies of the samples. Existing embedding-based methods ignore one crucial property: language-compositionality, i.e., a complex expression is built from its sub-structures. A compositional DA approach called TreeMix  ( Zhang et al., 2022d ) is developed to decompose sentences into their constituent sub-parts by constituency parsing tree. It then mixes them to generate new data, increasing the diversity and injecting compositionality into the model.

 
 
 

### 4.7. Sequence-to-Sequence

 
 Another critical task for text data is the sequence-to-sequence (Seq2Seq) task, which converts sequences from one domain (e.g., sentences in English) to sequences in another domain (e.g., the same sentences translated to French).

 
 
 SeqMix  ( Guo et al., 2020 ) is an effective MixDA algorithm for sequence-to-sequence tasks. It generates samples by softly mixing input-output sequence pairs with two binary combination vectors to encourage the neural model’s linear behavior. Another approach with the same name is proposed in work  ( Zhang et al., 2020b ) , which instead aims to improve the efficiency of active sequence labeling by expanding the queried data and synthesizing new sequences in each iteration. Specifically, SeqMix mixes queried samples in both the feature and target spaces. It guarantees plausibility via a discriminator that computes the perplexity scores for all the constructed candidates and returns the ones with low perplexity. For machine translation, MixDiversity  ( Li et al., 2021a ) is presented to generate different translations for the input sentence by linearly interpolating it with different sentence pairs sampled from the training corpus during decoding. Another work is multilingual crossover encoder-decoder (mXEncDec)  ( Cheng et al., 2022 ) , which interpolates instances from different language pairs into joint "crossover examples" to encourage sharing input and output spaces across languages. To handle the input perturbation issue in machine translation, AdvAug  ( Cheng et al., 2020 ) is designed to minimize the vicinal risk over virtual sentences sampled from two vicinity distributions, of which the crucial one is a novel vicinity distribution for adversarial sentences that describes a smooth interpolated embedding space centered around observed training sentence pairs. To improve the cross-lingual transfer ability, X-Mixup  ( Yang et al., 2022a ) is proposed to mix the representation of the source and target languages during training and inference to accommodate the representation discrepancy in the neural networks. For speech-to-text translation, Speech-TExt Manifold Mixup (STEMM)  ( Fang et al., 2022 ) is proposed to Mixup the representation sequences of both audio and text modalities, take both unimodal speech sequences and multimodal mixed sequences as input to the translation model in parallel, and regularize their output predictions with a self-learning framework.

 
 
 

### 4.8. Graph Neural Networks

 
 Graph neural networks (GNNs) have gained significant attention recently. While existing MixDA methods have shown promise in various domains, their direct application to graph data may not be suitable. This is primarily because it is challenging to determine how synthetic nodes should be connected to the original nodes through constructed edges while preserving the underlying graph topology.

 
 
 To handle this problem, GraphMix  ( Verma et al., 2021 ) is presented to train an additional fully connected network (FCN) to augment data, and only node features are fed into the FCN. Compared to conventional GNNs that only aggregate information from the neighbors in a few hops due to the over-smoothing issue, GraphMix conducts interpolation on randomly selected nodes, breaking the neighborhood limitation of GNNs and thus improving performance. Motivated by GraphMix, Graph Mixed Random Network Based on PageRank (PMRGNN)  ( Ma et al., 2022 ) expands neighborhood size for the random walk based graph neural networks. To combine both feature and structure information, NodeAug  ( Xue et al., 2021 ) is proposed with three variants: the first is NodeAug-I, which only mixes nodes’ features and ignores the topology; the second is NodeAug-N, which derives neighbor aggregation of virtual nodes by randomly choosing neighbors of two combined nodes with probability λ \lambda and ( 1 − λ ) (1-\lambda) , and then appending edges from these selected nodes; the last is NodeAug-S which scales all related source features accordingly, i.e., adding edges from all neighbors of the combined node to the mixed node. Exploiting the similarity between the neighborhood and receptive field sub-graph, Wang et al. (2021) present Mixup-based methods for node and graph classification tasks. Two-branch graph convolution is exploited in the node classification task to interpolate the irregular graph topology and mix their receptive fields. The interpolation is conducted on the aggregated representations from the two branches after each convolution layer. As for graph classification, Mixup performs in the semantic space. Another work, GraphMixup  ( Wu et al., 2021 ) , is designed for class-imbalanced node classification with 3 3 sub-modules: (i) feature Mixup performs in a constructed semantic relation space; (ii) two context-based self-supervised strategies are developed to model global and local structure information and conduct edge Mixup; and (iii) a reinforcement Mixup scheme adaptively determines the number of combined nodes. 𝒢 \mathcal{G} -Mixup  ( Han et al., 2022 ) generates synthetic graphs by interpolating sampled graphons in the Euclidean space, a generator estimated for each class. To create more plausible samples, Graph Transplant  ( Park et al., 2022a ) explicitly uses the node saliency information to guide the selection of sub-graphs and the label generation. Besides, mix-based schemes are also utilized to directly mix node embeddings for better performance in few-label scenarios  ( Zhao et al., 2022 ) .

 
 
 MixDA serves as an effective means to circumvent the over-smoothing issue by mixing distant nodes. Overall, the most critical point for MixDA on graph learning is how to use structure information to compose new data. Solutions are available to mix in the embedding space and combine with the help of sub-graphs or receptive fields. Moreover, directly creating new nodes and their edges using MixDA in a principled manner remains an open challenge.

 
 
 

### 4.9. Other Applications

 
 Metric Learning. Metric learning (ML) aims to quantify the similarity between examples and obtain an optimal task-specific distance metric, ensuring that embeddings of the same or similar classes are close while dissimilar ones are pushed far apart. On the one hand, some works (e.g., HDML ( Zheng et al., 2019 ) ) have demonstrated that producing new data points and training with metric learning loss can enhance generalization. However, these methods require additional generative networks for data augmentation, increasing the model size and training cost. On the other hand, the loss function of metric learning is based on two or more instances, sharing the same principle with MixDA approaches. Therefore, integrating mixing training into metric learning is preferred. Embedding Expansion ( Ko and Gu, 2020 ) synthesizes new data by mingling two or multiple feature representations. It also exploits negative pairs to learn the most discriminative feature. Similarly, Metric Mix (Metrix) ( Venkataramanan et al., 2022c ) adapts Mixup ( Zhang et al., 2018 ) and Manifold Mixup  ( Verma et al., 2019a ) into metric learning. It validates the effect of mixed samples with a novel metric called utilization and explores the regions of the embedding space beyond the training classes to refine the representation.

 
 
 Point Cloud. Directly applying MixDA to point cloud data may be challenging due to the lack of one-to-one correspondence between the points of two objects. Chen et al. (2020a) formulate data augmentation on point cloud as a shortest path linear interpolation problem and present PointMixup to construct new samples by assigning an optimal path function for two point clouds, resulting in a linear and invariant interpolation. Another work is PA-AUG  ( Choi et al., 2021 ) , which splits the objects into multiple sub-parts according to intra-object partition locations and then takes advantage of several MixDA methods in a partition-based way. To maintain topology information of the point cloud samples, Rigid Subset Mix (RSMix)  ( Lee et al., 2021a ) constructs synthetic examples by replacing an area of one sample with a shape-preserved part from the other sample. This extraction without distortion uses a carefully designed neighboring function considering the point cloud’s unordered structure and non-grid properties, thus preserving the structural information of the point cloud data.

 
 
 Potpourri. MixDA methods have also been widely applied in various other fields, such as federated learning  ( Yoon et al., 2021b ) , vision-language pre-training  ( Wang et al., 2022 ) , semantic segmentation  ( Kim et al., 2023 ) , opinion mining  ( Miao et al., 2020 ) , speaker verification  ( Zhang et al., 2022b ) , speech recognition  ( Meng et al., 2021 ) , video classification  ( Yun et al., 2020 ) , meta-learning  ( Yao et al., 2021 ; Chen et al., 2021a ) , knowledge distillation  ( Xu et al., 2023 ) , collaborative filtering  ( Moon et al., 2023 ) and factorization machines  ( Wu et al., 2024 ) .

 
 
 

### 4.10. Discussion

 
 The exploration of MixDA techniques across various data modalities, including image, text, audio, point cloud, and graph, necessitates an understanding of their suitability relative to the inherent characteristics of each modality.

 
 
 Image. The initial developments in MixDA, specifically Mixup and Cutmix, focused on image classification tasks, setting a precedent for their extensive application in image data. Consequently, MixDA techniques are predominantly applied and highly effective across various image-related tasks.

 
 
 Text. Text data, with its discrete and sequential nature, presents unique challenges and opportunities for MixDA application. Cutmix-based approaches are effective; they allow for the practical manipulation of text at both the data and embedding levels. For instance, cutting and pasting segments of sentences is straightforward. Conversely, Mixup-based methods at the data level are less effective due to the potential generation of non-existent words. However, these issues can be mitigated by applying Mixup at the embedding level.

 
 
 Audio. Audio shares certain analytical similarities with text data, allowing for successfully applying Mixup-based and Cutmix-based MixDA methods. Traditional Mixup, for instance, when applied to audio, results in a perceivable overlay of sounds, akin to overhearing simultaneous conversations, which maintains the comprehensibility of the data.

 
 
 Point Cloud. The application of MixDA to point cloud is challenging due to the lack of direct correspondence between objects. While direct application of original Mixup and Cutmix is problematic, adaptations such as mixing at the embedding level or employing transformations like optimal path functions and recombining object parts prove viable.

 
 
 Graph. Graphs, representing non-Euclidean structures with crucial topological information, require nuanced approaches for data mixing. The complexity of determining labels for mixed samples necessitates domain-specific knowledge, making embedding-level mixing a preferred method. When mixing at the node level, a significant challenge lies in determining connectivity for newly generated nodes.

 
 
 MixDA’s flexibility makes it applicable across diverse data modalities. While certain modalities pose specific challenges, the core concept of interpolating inputs and targets remains universal. Furthermore, embedding-level mixing presents a consistently effective strategy. Future research will need to address the complexities associated with non-conventional data types such as graph and point cloud, continuing to refine and expand the applicability of MixDA.

 
 
 
 

## 5. Explainability Analysis of MixDA

 
 Although numerous MixDA methods have been successfully used to solve a range of applications, the underlying reasons for their effectiveness remain unclear. In this section, we systematically review the explainability foundations of MixDA, focusing on three different aspects that explain why mixed samples aid generalization: vicinal risk minimization (VRM)  ( Zhang et al., 2018 ; Lim et al., 2022 ; Pinto et al., 2022 ; Mangla et al., 2020 ) , mode regularization  ( Carratino et al., 2022 ; Liang et al., 2018 ; Zhang et al., 2021a ; Park et al., 2022c ) , and uncertainty calibration  ( Zhang et al., 2022a ; Thulasidasan et al., 2019 ; Zhang et al., 2020a ) . We also provide some interpretations of why MixDA works well.

 
 

### 5.1. Vicinal Risk Minimization

 
 Supervised learning aims to find out a mapping function f f in the hypothesis space ℱ \mathcal{F} that models the relationship between the input random variable 𝐗 \mathbf{X} and output random variable 𝐘 \mathbf{Y} following a joint distribution P ⁡ ( 𝐗 , 𝐘 ) P(\mathbf{X},\mathbf{Y}) . To achieve this goal, a loss function ℒ \mathcal{L} is defined as the discrepancy between the model prediction f ⁡ ( 𝐱 ) f(\mathbf{x}) and the ground truth 𝐲 \mathbf{y} for the sample ( 𝐱 , 𝐲 ) ∼ P (\mathbf{x},\mathbf{y})\sim P . An optimization algorithm is required to minimize the average of the loss function ℒ \mathcal{L} over the joint distribution P P to obtain the optimal function f ∗ f^{*} :

 

 
 (29) | 
 | 
 f ∗ = arg ⁡ min ⁡ ∫ f ∈ ℱ ⁡ ℒ ⁡ ( f ⁡ ( 𝐱 ) , 𝐲 ) ​ 𝑑 P ​ ( 𝐱 , 𝐲 ) . f^{*}=\arg\min_{f\in\mathcal{F}}\int\mathcal{L}(f(\mathbf{x}),\mathbf{y})\mathrm{d}P(\mathbf{x},\mathbf{y}). | 
 | 
 

 Unfortunately, in most cases, the joint distribution P P is unknown. As a remedy, the prevailing practice is to collect some training data { ( 𝐱 i , 𝐲 i ) } i = 1 N \{(\mathbf{x}_{i},\mathbf{y}_{i})\}_{i=1}^{N} , where N N is the number of training examples and ( 𝐱 i , 𝐲 i ) ∼ P (\mathbf{x}_{i},\mathbf{y}_{i})\sim P , and utilize empirical risk to approximate expected risk, which is then minimized to attain the optimum, i.e., empirical risk minimization (ERM):

 

 
 (30) | 
 | 
 P ⁡ ( 𝐱 , 𝐲 ) ≈ P ψ ​ ( 𝐱 , 𝐲 ) = 1 N ​ ∑ i = 1 N ρ ⁡ ( 𝐱 = 𝐱 i , 𝐲 = 𝐲 i ) , P(\mathbf{x},\mathbf{y})\approx P_{\psi}(\mathbf{x},\mathbf{y})=\frac{1}{N}\sum_{i=1}^{N}\rho(\mathbf{x}=\mathbf{x}_{i},\mathbf{y}=\mathbf{y}_{i}), | 
 | 
 

 where ρ ⁡ ( 𝐱 = 𝐱 i , 𝐲 = 𝐲 i ) \rho(\mathbf{x}=\mathbf{x}_{i},\mathbf{y}=\mathbf{y}_{i}) is a Dirac mass centered at ( 𝐱 i , 𝐲 i ) (\mathbf{x}_{i},\mathbf{y}_{i}) . One of the major concerns for ERM is its generalization performance. When the size of the hypothesis space (measured by the number of the model’s parameters) is comparable to or larger than the number of training data N N , the obtained model via ERM is prone to memorizing training samples and, consequently, performs poorly when encountering new data. The reason is that the support of ρ ⁡ ( 𝐱 ) \rho(\mathbf{x}) is a one-point set { 𝐱 i } i = 1 N \{\mathbf{x}_{i}\}_{i=1}^{N} , therefore, P ψ ​ ( 𝐱 , 𝐲 ) P_{\psi}(\mathbf{x},\mathbf{y}) cannot approximate P ⁡ ( 𝐗 , 𝐘 ) P(\mathbf{X},\mathbf{Y}) exactly. To address this problem, vicinal risk minimization (VRM)  ( Chapelle et al., 2000 ) is proposed to improve the viability of ERM by replacing the Dirac mass with a vicinity function:

 

 
 (31) | 
 | 
 P ( 𝐱 , 𝐲 ) ≈ P 𝒱 ( 𝐱 ~ , 𝐲 ~ ) = 1 N ∑ i = 1 N 𝒱 ( 𝐱 ~ , 𝐲 ~ ∣ 𝐱 i , 𝐲 i ) , P(\mathbf{x},\mathbf{y})\approx P_{\mathcal{V}}(\tilde{\mathbf{x}},\tilde{\mathbf{y}})=\frac{1}{N}\sum_{i=1}^{N}\mathcal{V}(\tilde{\mathbf{x}},\tilde{\mathbf{y}}\mid\mathbf{x}_{i},\mathbf{y}_{i}), | 
 | 
 

 where 𝒱 \mathcal{V} is the vicinity function that gauges the probability of the virtual example ( 𝐱 ~ , 𝐲 ~ ) (\tilde{\mathbf{x}},\tilde{\mathbf{y}}) appears in the vicinity of the training sample ( 𝐱 i , 𝐲 i ) (\mathbf{x}_{i},\mathbf{y}_{i}) . In this vein, Mixup and Cutmix can be reformulated as a group of generic vicinal distribution:

 

 
 | 
 𝒱 Mixup = 1 N ​ ∑ j N 𝔼 𝜆 \displaystyle\mathcal{V}_{\text{Mixup}}=\frac{1}{N}\sum_{j}^{N}\underset{\lambda}{\mathbb{E}} | 
 [ ρ ⁡ ( 𝐱 ~ = λ ​ 𝐱 i + ( 1 − λ ) ​ 𝐱 j , 𝐲 ~ = λ ​ 𝐲 i + ( 1 − λ ) ​ 𝐲 j ) ] , \displaystyle[\rho(\tilde{\mathbf{x}}=\lambda\mathbf{x}_{i}+(1-\lambda)\mathbf{x}_{j},\tilde{\mathbf{y}}=\lambda\mathbf{y}_{i}+(1-\lambda)\mathbf{y}_{j})], | 
 | 
 
 
 (32) | 
 | 
 𝒱 Cutmix = 1 N ​ ∑ j N 𝔼 𝜆 \displaystyle\mathcal{V}_{\text{Cutmix}}=\frac{1}{N}\sum_{j}^{N}\underset{\lambda}{\mathbb{E}} | 
 [ ρ ⁡ ( 𝐱 ~ = 𝐌 ⊙ 𝐱 i + ( 1 − 𝐌 ) ⊙ 𝐱 j , 𝐲 ~ = λ ​ 𝐲 i + ( 1 − λ ) ​ 𝐲 j ) ] . \displaystyle[\rho(\tilde{\mathbf{x}}=\mathbf{M}\odot\mathbf{x}_{i}+(1-\mathbf{M})\odot\mathbf{x}_{j},\tilde{\mathbf{y}}=\lambda\mathbf{y}_{i}+(1-\lambda)\mathbf{y}_{j})]. | 
 | 
 

 
 
 

### 5.2. Model Regularization

 
 From the perspective of model regularization, MixDA methods aim to minimize the standard empirical risk on transformed data with a class of specific perturbations  ( Carratino et al., 2022 ) . Inspired by previous analyses of dropout  ( Wager et al., 2013 ; Wei et al., 2020a ) , a regularized objective is derived to specify the regularization effects of blended examples  ( Carratino et al., 2022 ) . It has been demonstrated that Mixup and Cutmix enjoy benefits similar to dropout  ( Srivastava et al., 2014 ) and label smoothing  ( Pereyra et al., 2017 ) . Meanwhile, this work proposes to interpret them to smooth the Jacobian of the model and upgrade the calibration. It is found that the decision surface trained with mingled samples is smoother than that obtained by conventional ERM  ( Liang et al., 2018 ) . Another parallel and independent work  ( Zhang et al., 2021a ) demonstrates that training with mixed samples is the same as approximating the regularized loss minimization. Specifically, mix-based approaches directly restrain the Rademacher Complexity  ( Bartlett and Mendelson, 2002 ) of the underlying model and give concrete generalization error bounds. Therefore, it can address the issue of overfitting to some extent. Besides, Mixed sample data augmentation is proven as a pixel-level regularization exposed on the input gradients and Hessians  ( Park et al., 2022c ) . For example, Cutmix actually tries to regularize the input gradients based on pixel distances.

 
 
 

### 5.3. Uncertainty Calibration

 
 Let us consider the most commonly used calibration metric ECE (expected calibration error):

 

 
 (33) | 
 | 
 ECE = 𝔼 p ∼ P p ^ ​ [ | ℙ ⁡ ( y ^ = y ∣ p ^ = p ) − p | ] , \text{ECE}=\mathbb{E}_{p\sim P_{\hat{p}}}[|\mathbb{P}(\hat{y}=y\mid\hat{p}=p)-p|], | 
 | 
 

 where P p ^ P_{\hat{p}} is the probability distribution of p ^ \hat{p} – the largest item in the model output softmax vector. Following  ( Zhang et al., 2022a ) , we assume the model is a Gaussian classifier:

 

 
 (34) | 
 | 
 f ⁡ ( 𝐱 ) = sgn ⁡ ( ω T ​ 𝐱 ) , f(\mathbf{x})=\operatorname{sgn}(\mathbf{\omega}^{\mathrm{T}}\mathbf{x}), | 
 | 
 

 where ω = ∑ i = 1 N 𝐱 i ​ y i / N \mathbf{\omega}=\sum_{i=1}^{N}\mathbf{x}_{i}y_{i}/N . Note that y i y_{i} is the scalar label ( y i ∈ { − 1 , + 1 } y_{i}\in\{-1,+1\} ), which is different from 𝐲 i \mathbf{y}_{i} is a one-hot vector. Given 𝐱 \mathbf{x} and ω \mathbf{\omega} , the prediction is obtained by:

 

 
 (35) | 
 | 
 y = f ⁡ ( 𝐱 ) = arg ⁡ max c ∈ { − 1 , 1 } ​ p c ​ ( 𝐱 ) , y=f(\mathbf{x})=\arg\max_{c\in\{-1,1\}}p_{c}(\mathbf{x}), | 
 | 
 

 where p c ​ ( 𝐱 ) p_{c}(\mathbf{x}) is the probability of class c c and is defined as:

 

 
 (36) | 
 | 
 p c ​ ( 𝐱 ) = 1 e − 2 c ⋅ ω T 𝐱 / σ 2 + 1 , p_{c}(\mathbf{x})=\frac{1}{e^{-2c\cdot\mathbf{\omega}^{\mathrm{T}}\mathbf{x}/\mathbf{\sigma}^{2}}+1}, | 
 | 
 

 where σ \mathbf{\sigma} is the standard deviation vector. After applying MixDA (taking Mixup as an example), augmented data is { 𝐱 ~ i , j ​ ( λ ) , y ~ i , j ​ ( λ ) } i , j = 1 N \{\tilde{\mathbf{x}}_{i,j}(\lambda),\tilde{y}_{i,j}(\lambda)\}_{i,j=1}^{N} , leading to another classifier:

 

 
 (37) | 
 | 
 ω mix = 𝔼 λ ∼ P λ ​ ∑ i , j = 1 N 𝐱 ~ i , j ​ ( λ ) ​ y ~ i , j ​ ( λ ) / N 2 , f mix ​ ( 𝐱 ) = sgn ⁡ ( ω mix T ​ 𝐱 ) , \mathbf{\omega}_{\text{mix}}=\mathbb{E}_{\lambda\sim P_{\lambda}}\sum_{i,j=1}^{N}\tilde{\mathbf{x}}_{i,j}(\lambda)\tilde{y}_{i,j}(\lambda)/N^{2},\ f_{\text{mix}}(\mathbf{x})=\operatorname{sgn}(\mathbf{\omega}_{\text{mix}}^{\mathrm{T}}\mathbf{x}), | 
 | 
 

 where P λ P_{\lambda} is the distribution of the mix ratio. The result shows that MixDA helps calibration more when the feature dimension is higher: ECE ​ ( f mix ) ECE ​ ( f ) \text{ECE}(f_{\text{mix}}) \text{ECE}(f) . Besides, the above analysis also holds for semi-supervised learning  ( Zhang et al., 2022a ) .

 
 
 Experiments on several image classification benchmarks and models demonstrate that mix-trained deep neural networks can significantly improve calibration  ( Thulasidasan et al., 2019 ) . In other words, the generated softmax scores are much closer to the actual likelihood than the conventional models. Besides, solely mingling inputs or features cannot achieve the same degree of calibration. This suggests that the mix of targets, as a form of label smoothing  ( Szegedy et al., 2016 ) , plays a vital role in improving calibration  ( Thulasidasan et al., 2019 ) . In summary, mixed training effectively mitigates over-confidence in data with noise or from out-of-distribution. Moreover, by incorporating Mixup inference, models can be trained using the original one-hot labels, thereby mitigating the negative impact of the confidence penalty  ( Wang et al., 2023 ) .

 
 
 

### 5.4. Properties Interpretations of MixDA

 
 MixDA possesses numerous appealing properties that have been extensively surveyed in this work. However, only a few works have explicitly summarized these properties and interpreted their functions in improving the model performance. In this subsection, we aim to bridge this gap.

 
 
 
 • 
 
 MixDA methods such as Mixup and Cutmix oblige linear behavior between training examples. This property can significantly decrease the probability of oscillations when predicting examples that are not from the training distribution. It also facilitates model generalization by averting memorization.

 

 • 
 
 As a kind of "local linearity" regularization, MixDA training encourages the decision boundaries to transit linearly between classes and smooths the estimate of uncertainty  ( Verma et al., 2019a ; Venkataramanan et al., 2022a ; Lim et al., 2022 ) . Through the lens of label smoothing  ( Szegedy et al., 2016 ) , manipulating mixing targets with the ratio λ : ( 1 − λ ) \lambda:(1-\lambda) prevents the excessive pursuit of the rigid 0 0 - 1 1 estimation. It, in turn, enables the model to account for uncertainty. These merits also boost the calibration and performance in unbalanced scenarios (e.g., in positive and unlabeled learning  ( Wei et al., 2020b ; Li et al., 2022b ) ).

 

 • 
 
 Training with mixed instances is more stable regarding gradient norms and model predictions, which is crucial for generative models such as GANs  ( Goodfellow et al., 2014 ) and DDPM  ( Ho et al., 2020 ) . Furthermore, MixDA bears some similarities to adversarial training, as both aim to explore areas outside the data manifold. Additionally, as a typical data augmentation method, MixDA alleviates the data-hungry issue in adversarial training  ( Schmidt et al., 2018 ) .

 

 • 
 
 Both Mixup and Cutmix methods are simple enough to be incorporated into existing learning models. More importantly, mix operations are usually data-independent and model-agnostic. Consequently, MixDA methods are generic and applicable to various domains (cf. Section  4 ).

 

 
 
 
 
 

## 6. Discussion

 
 In this section, we present findings w.r.t. the current research on MixDA and provide insights into the remaining open challenges in this domain. By doing so, future researchers can pinpoint promising future research directions.

 
 

### 6.1. Revisiting MixDA

 
 By revisiting current MixDA methods, we have the following important findings.

 
 • 
 
 Finding 1: Research attention on Mixup and Cutmix. Despite the similarities between Mixup and Cutmix, their adaptations focus on different aspects. For Mixup, considerable works aim to adaptively determine the mix ratio λ \lambda (cf. Section  3.1.3 ) or choose appropriate samples for mixing (cf. Section  3.1.4 ). In contrast, existing methods for Cutmix focus on studying how to select the cut patch and its location for pasting (cf. Section  3.2.2 and Section  3.2.3 ). This difference stems from the fact that Cutmix was proposed from a local perspective with the constructed rectangle region, while Mixup was designed from a holistic view with a global mix ratio.

 

 • 
 
 Finding 2: Tradeoff between plausibility and diversity. MixDA has undergone an evolutionary process with respect to plausibility and diversity. Initially, vanilla Mixup and Cutmix blindly combined training samples. Later, saliency information was leveraged to guide the combination of mixed examples (cf. Section  3.1.5 and Section  3.2.2 ), aiming to generate plausible data. However, the informative patches are limited, resulting in less diverse generated data. Conversely, overemphasizing diversity can lead to many irrational augmented data. Therefore, future work should focus on achieving a favorable tradeoff between these two factors.

 

 • 
 
 Finding 3: Mixing more samples and the mix of MixDA. Most reviewed methods consider how to mix two training samples, while some researchers have started investigating how to combine more examples in each mix process to increase augmentation diversity (cf. Section  3.1.6 ). Additionally, combining MixDA methods, also known as the mix of MixDA, has gained popularity and has been demonstrated to be effective in further enhancing model performance (cf. Section  3.3 ).

 

 
 
 
 

### 6.2. Open Challenge Problems

 
 Despite the desirable performance of MixDA methods in various applications, several challenges persist in training MixDA models. We outline these challenges as follows:

 
 • 
 
 Challenge 1: Distortion in Mix. MixDA methods inevitably introduce some distortion to the original inputs, potentially causing a mismatch between the vicinal and actual data distribution. While this property benefits robustness, mixed examples also introduce noise into the data manifold. Consequently, a critical challenge in the MixDA domain is how to reduce or leverage this perturbation effectively.

 

 • 
 
 Challenge 2: Combining multiple DA. Integrating various data augmentation methods into a pipeline is a natural approach for deep learning methods. However, determining the order of multiple data augmentation approaches poses a challenge. For example, it is widely believed that applying Mixup after rotation yields better results than applying rotation after Mixup. Unfortunately, a principled method and theoretical understanding of optimal ordering remain elusive in the literature.

 

 • 
 
 Challenge 3: The function of λ \lambda . The function of the mix ratio λ \lambda is not well understood. Conventional Mixup and Cutmix strategies determine their value empirically using a Beta distribution. Although some research has developed methods to determine this important coefficient adaptively, few studies elucidate the implications behind this ratio and how to determine its value efficiently.

 

 
 
 
 

### 6.3. Research Opportunity

 
 Finally, we outline several exciting research opportunities for future researchers in MixDA.

 
 
 
 • 
 
 Investigating the relationship between MixDA and regularization. Further investigation is required to explore the relationship between MixDA and regularization. For example, Mixup training requires a significantly lower weight decay on CIFAR-10  ( Krizhevsky et al., 2009 ) , demonstrating that MixDA has cross-cutting and complementary effects with regularization. Besides, Reformulating MixDA as a form of regularization can benefit both lines of work. For instance, exploring the relationship between MixDA and Lipschitz continuity is a potential research proposal.

 

 • 
 
 Exploring mix-based test-time augmentation for uncertainty estimation and calibration. Although mix training has demonstrated effectiveness in uncertainty estimation and calibration, using mix-based test-time augmentation for these purposes has been less explored. For example, generating test-time augmented versions for each testing sample by mixing it with examples randomly sampled from the training data is a straightforward solution. However, maintaining a buffer for all training data may be prohibitive. Therefore, finding prototypes or proxies for training examples is an exciting problem that requires further study.

 

 • 
 
 Integrating MixDA with the large model. Large models have recently received significant attention and discussion; therefore, using MixDA for large language models is an exciting research direction. First, the few-shot learning capability of large language models provides new application scenarios for MixDA. By interpolating and mixing a small number of samples, the generalization and robustness of the large language model can be further improved. For example, in the in-context learning paradigm of large language models, exploring whether mixing prompts of different tasks can improve the model’s task transfer ability is an exciting research direction. Second, regarding large visual language models, image and text features are mixed to alleviate the differences in multimodal data. Moreover, the rich knowledge learned from the large model can also guide the MixDA process at the semantic level, such as preventing the generation of unreasonable mixed samples. Last, MixDA may help improve the security and fairness of large models. By generating hard examples, Mixup exposes models to more diverse adversarial cases, thereby reducing bias and vulnerabilities. Proper feature mixing is also expected to mitigate model bias in tasks involving sensitive attributes.

 

 • 
 
 Applying MixDA to time series and reinforcement learning tasks. Extending MixDA methods to time series or reinforcement learning tasks, such as generating scenarios for autonomous driving, presents a pressing challenge due to temporal dependence and the cumulative error issue. A possible avenue is interpolating perceptions from adjacent timestamps to provide more fine-grained information for downstream tasks.

 

 • 
 
 Exploring mixing more than two examples. The field of combining more than two examples in MixDA is still under-explored. This area of research can potentially significantly increase the diversity of augmented data. Iteratively executing Mixup and Cutmix to combine multiple samples can improve the diversity of augmented data and leverage the advantages of Mixup’s global view and Cutmix’s local perspective.

 

 • 
 
 Identifying the limits of MixDA. Identifying the limits of MixDA methods is essential for understanding existing approaches and designing improved ones. For example, when applying MixDA to graph learning, the generated soft labels for the newly constructed nodes can exacerbate the under-confidence issue in GNNs.

 

 
 
 
 
 

## 7. Conclusion

 
 Data augmentation has consistently been an important research topic in machine learning and deep learning. In this survey, we systematically review the mix-based data augmentation methods by providing an in-depth analysis of techniques, benchmarks, applications, and theoretical foundations. First, we introduce a new classification for MixDA methods. In this context, we present a more fine-grained taxonomy that categorizes existing MixDA approaches into different groups based on their motivations. Then, we thoroughly review various MixDA methods while recapping their advantages and disadvantages. Besides, we comprehensively survey more than eight applications of MixDA. Furthermore, we provide theoretical examinations of MixDA through the lens of VRM, model regularization, and uncertainty calibration while explaining the success of MixDA by examining its critical properties. Finally, we summarize our important findings regarding the trends in MixDA research, present the main challenges in existing studies, and outline potential research opportunities for future work in this field.

 
 
 

## References

 
 
 Anaby-Tavor et al . (2020) 
 
Ateret Anaby-Tavor, Boaz Carmeli, Esther Goldbraich, Amir Kantor, George Kour, Segev Shlomov, Naama Tepper, and Naama Zwerdling. 2020.

 
 Do not have enough data? Deep learning to the rescue!. In Proceedings of the AAAI conference on artificial intelligence . 7383–7390.

 
 
 

 
 Archambault et al . (2019) 
 
Guillaume P Archambault, Yongyi Mao, Hongyu Guo, and Richong Zhang. 2019.

 
 Mixup as directional adversarial training.

 
 (2019).

 
 arXiv:1906.06875

 

 
 Asuncion and Newman (2007) 
 
Arthur Asuncion and David Newman. 2007.

 
 UCI machine learning repository.

 
 
 
 
 

 
 Baek et al . (2021) 
 
Kyungjune Baek, Duhyeon Bang, and Hyunjung Shim. 2021.

 
 GridMix: Strong regularization through local context mapping.

 
 Pattern recognition 109 (2021), 107594.

 
 
 

 
 Baena et al . (2022) 
 
Raphael Baena, Lucas Drumetz, and Vincent Gripon. 2022.

 
 Preventing manifold intrusion with locality: Local Mixup.

 
 (2022).

 
 arXiv:2201.04368

 

 
 Bartlett and Mendelson (2002) 
 
Peter L Bartlett and Shahar Mendelson. 2002.

 
 Rademacher and Gaussian complexities: Risk bounds and structural results.

 
 Journal of Machine Learning Research 3, Nov (2002), 463–482.

 
 
 

 
 Bayer et al . (2023) 
 
Markus Bayer, Marc-André Kaufhold, Björn Buchhold, Marcel Keller, Jörg Dallmeyer, and Christian Reuter. 2023.

 
 Data augmentation in natural language processing: a novel text generation approach for long and short text classifiers.

 
 International Journal of Machine Learning and Cybernetics 14, 1 (2023), 135–150.

 
 
 

 
 Bayer et al . (2022) 
 
Markus Bayer, Marc-André Kaufhold, and Christian Reuter. 2022.

 
 A survey on data augmentation for text classification.

 
 Comput. Surveys 55, 7 (2022), 1–39.

 
 
 

 
 Beckham et al . (2019) 
 
Christopher Beckham, Sina Honari, Vikas Verma, Alex M Lamb, Farnoosh Ghadiri, R Devon Hjelm, Yoshua Bengio, and Chris Pal. 2019.

 
 On adversarial Mixup resynthesis. In Advances in neural information processing systems . 4348–4359.

 
 
 

 
 Berthelot et al . (2019a) 
 
David Berthelot, Nicholas Carlini, Ekin D Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang, and Colin Raffel. 2019a.

 
 Remixmatch: Semi-supervised learning with distribution alignment and augmentation anchoring.

 
 (2019).

 
 arXiv:1911.09785

 

 
 Berthelot et al . (2019b) 
 
David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A Raffel. 2019b.

 
 Mixmatch: A holistic approach to semi-supervised learning. In Advances in neural information processing systems . 5050–5060.

 
 
 

 
 Berthelot et al . (2019c) 
 
David Berthelot, Colin Raffel, Aurko Roy, and Ian Goodfellow. 2019c.

 
 Understanding and improving interpolation in autoencoders via an adversarial regularizer. In International conference on learning representations .

 
 
 

 
 Bunk et al . (2021) 
 
Jason Bunk, Srinjoy Chattopadhyay, BS Manjunath, and Shivkumar Chandrasekaran. 2021.

 
 Adversarially optimized Mixup for robust classification.

 
 (2021).

 
 arXiv:2103.11589

 

 
 Byun et al . (2023) 
 
Junyoung Byun, Myung-Joon Kwon, Seungju Cho, Yoonji Kim, and Changick Kim. 2023.

 
 Introducing competition to boost the transferability of targeted adversarial examples through clean feature Mixup. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 24648–24657.

 
 
 

 
 Carratino et al . (2022) 
 
Luigi Carratino, Moustapha Cissé, Rodolphe Jenatton, and Jean-Philippe Vert. 2022.

 
 On mixup regularization.

 
 Journal of machine learning research 23, 1 (2022), 325:1–325:31.

 
 
 

 
 Cascante-Bonilla et al . (2021) 
 
Paola Cascante-Bonilla, Arshdeep Sekhon, Yanjun Qi, and Vicente Ordonez. 2021.

 
 Evolving image compositions for feature representation learning. In Proceedings of the british machine vision conference . 199.

 
 
 

 
 Chapelle et al . (2000) 
 
Olivier Chapelle, Jason Weston, Léon Bottou, and Vladimir Vapnik. 2000.

 
 Vicinal risk minimization. In Advances in neural information processing systems . 416–422.

 
 
 

 
 Chawla et al . (2002) 
 
Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. 2002.

 
 SMOTE: Synthetic minority over-sampling technique.

 
 Journal of Artificial Intelligence Research 16 (2002), 321–357.

 
 
 

 
 Chen et al . (2021b) 
 
Chen Chen, Jingfeng Zhang, Xilie Xu, Tianlei Hu, Gang Niu, Gang Chen, and Masashi Sugiyama. 2021b.

 
 Guided interpolation for adversarial training.

 
 (2021).

 
 arXiv:2102.07327

 

 
 Chen et al . (2022a) 
 
John Chen, Samarth Sinha, and Anastasios Kyrillidis. 2022a.

 
 StackMix: A complementary mix algorithm. In Uncertainty in artificial intelligence . 326–335.

 
 
 

 
 Chen et al . (2020b) 
 
Jiaao Chen, Zhenghui Wang, Ran Tian, Zichao Yang, and Diyi Yang. 2020b.

 
 Local additivity based data augmentation for semi-supervised NER. In Proceedings of the conference on empirical methods in natural language processing . 1241–1251.

 
 
 

 
 Chen et al . (2020c) 
 
Jiaao Chen, Zichao Yang, and Diyi Yang. 2020c.

 
 MixText: Linguistically-informed interpolation of hidden space for semi-Supervised text classification. In Proceedings of the annual meeting of the association for computational linguistics . 2147–2157.

 
 
 

 
 Chen et al . (2022b) 
 
Jie-Neng Chen, Shuyang Sun, Ju He, Philip HS Torr, Alan Yuille, and Song Bai. 2022b.

 
 Transmix: Attend to mix for vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 12125–12134.

 
 
 

 
 Chen and He (2021) 
 
Xinlei Chen and Kaiming He. 2021.

 
 Exploring simple Siamese representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 15750–15758.

 
 
 

 
 Chen et al . (2020a) 
 
Yunlu Chen, Vincent Tao Hu, Efstratios Gavves, Thomas Mensink, Pascal Mettes, Pengwan Yang, and Cees GM Snoek. 2020a.

 
 PointMixup: Augmentation for point clouds. In European conference on computer vision . 330–345.

 
 
 

 
 Chen et al . (2021a) 
 
Yangbin Chen, Yun Ma, Tom Ko, Jianping Wang, and Qing Li. 2021a.

 
 MetaMix: Improved meta-learning with interpolation-based consistency regularization. In International conference on pattern recognition . 407–414.

 
 
 

 
 Cheng et al . (2022) 
 
Yong Cheng, Ankur Bapna, Orhan Firat, Yuan Cao, Pidong Wang, and Wolfgang Macherey. 2022.

 
 Multilingual mix: Example interpolation improves multilingual neural machine translation. In Proceedings of the annual meeting of the association for computational linguistics . 4092–4102.

 
 
 

 
 Cheng et al . (2020) 
 
Yong Cheng, Lu Jiang, Wolfgang Macherey, and Jacob Eisenstein. 2020.

 
 AdvAug: Robust adversarial augmentation for neural machine translation. In Proceedings of the annual meeting of the association for computational linguistics . 5961–5970.

 
 
 

 
 Chidambaram et al . (2022) 
 
Muthu Chidambaram, Xiang Wang, Yuzheng Hu, Chenwei Wu, and Rong Ge. 2022.

 
 Towards understanding the data dependency of Mixup-style training. In International conference on learning representations .

 
 
 

 
 Choi et al . (2022) 
 
Hyeong Kyu Choi, Joonmyung Choi, and Hyunwoo J Kim. 2022.

 
 Tokenmixup: Efficient attention-guided token-level data augmentation for transformers. In Advances in Neural Information Processing Systems . 14224–14235.

 
 
 

 
 Choi et al . (2021) 
 
Jaeseok Choi, Yeji Song, and Nojun Kwak. 2021.

 
 Part-aware data augmentation for 3d object detection in point cloud. In IEEE/RSJ international conference on intelligent robots and systems . 3391–3397.

 
 
 

 
 Chou et al . (2020) 
 
Hsin-Ping Chou, Shih-Chieh Chang, Jia-Yu Pan, Wei Wei, and Da-Cheng Juan. 2020.

 
 Remix: Rebalanced Mixup. In European conference on computer vision . 95–110.

 
 
 

 
 Chu et al . (2020) 
 
Xiangxiang Chu, Xiaohang Zhan, and Xiaolin Wei. 2020.

 
 Beyond single instance multi-view unsupervised representation learning.

 
 (2020).

 
 arXiv:2011.13356

 

 
 Cubuk et al . (2019) 
 
Ekin D Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V Le. 2019.

 
 Autoaugment: Learning augmentation strategies from data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 113–123.

 
 
 

 
 Dabouei et al . (2021) 
 
Ali Dabouei, Sobhan Soleymani, Fariborz Taherkhani, and Nasser M Nasrabadi. 2021.

 
 Supermix: Supervising the mixing data augmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 13794–13803.

 
 
 

 
 DeVries and Taylor (2017) 
 
Terrance DeVries and Graham W Taylor. 2017.

 
 Improved regularization of convolutional neural networks with cutout.

 
 (2017).

 
 arXiv:1708.04552

 

 
 Dosovitskiy et al . (2020) 
 
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al . 2020.

 
 An image is worth 16x16 words: Transformers for image recognition at scale. In International conference on learning representations .

 
 
 

 
 Fang et al . (2022) 
 
Qingkai Fang, Rong Ye, Lei Li, Yang Feng, and Mingxuan Wang. 2022.

 
 STEMM: Self-learning with speech-text manifold Mixup for speech translation. In Proceedings of the annual meeting of the association for computational linguistics . 7050–7062.

 
 
 

 
 Faramarzi et al . (2022) 
 
Mojtaba Faramarzi, Mohammad Amini, Akilesh Badrinaaraayanan, Vikas Verma, and Sarath Chandar. 2022.

 
 Patchup: A regularization technique for convolutional neural networks. In Proceedings of the AAAI conference on artificial intelligence . 589–597.

 
 
 

 
 Feng et al . (2021b) 
 
Hao-Zhe Feng, Kezhi Kong, Minghao Chen, Tianye Zhang, Minfeng Zhu, and Wei Chen. 2021b.

 
 SHOT-VAE: Semi-supervised deep generative models with label-aware ELBO approximations. In Proceedings of the AAAI conference on artificial intelligence . 7413–7421.

 
 
 

 
 Feng et al . (2021a) 
 
Steven Y Feng, Varun Gangal, Jason Wei, Sarath Chandar, Soroush Vosoughi, Teruko Mitamura, and Eduard Hovy. 2021a.

 
 A survey of data augmentation approaches for NLP. In Findings of the association for computational linguistics: ACL/IJCNLP . 968–988.

 
 
 

 
 Goodfellow et al . (2014) 
 
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014.

 
 Generative adversarial nets. In Advances in neural information processing systems . 2672–2680.

 
 
 

 
 Goodfellow et al . (2015) 
 
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. 2015.

 
 Explaining and harnessing adversarial examples. In International conference on learning representations .

 
 
 

 
 Greenewald et al . (2023) 
 
Kristjan Greenewald, Anming Gu, Mikhail Yurochkin, Justin Solomon, and Edward Chien. 2023.

 
 k-Mixup regularization for deep learning via optimal transport.

 
 Transactions on machine learning research (2023).

 
 
 

 
 Guo et al . (2020) 
 
Demi Guo, Yoon Kim, and Alexander M Rush. 2020.

 
 Sequence-level mixed sample data augmentation. In Proceedings of the conference on empirical methods in natural language processing . 5547–5552.

 
 
 

 
 Guo (2020) 
 
Hongyu Guo. 2020.

 
 Nonlinear Mixup: Out-of-manifold data augmentation for text classification. In Proceedings of the AAAI conference on artificial intelligence . 4044–4051.

 
 
 

 
 Guo et al . (2019a) 
 
Hongyu Guo, Yongyi Mao, and Richong Zhang. 2019a.

 
 Augmenting data with Mixup for sentence classification: An empirical study.

 
 (2019).

 
 arXiv:1905.08941

 

 
 Guo et al . (2019b) 
 
Hongyu Guo, Yongyi Mao, and Richong Zhang. 2019b.

 
 Mixup as locally linear out-of-manifold regularization. In Proceedings of the AAAI conference on artificial intelligence . 3714–3722.

 
 
 

 
 Guo et al . (2021) 
 
Xiaoyang Guo, Tianhao Zhao, Yutian Lin, and Bo Du. 2021.

 
 MixSiam: A mixture-based approach to self-supervised representation learning.

 
 (2021).

 
 arXiv:2111.02679

 

 
 Hammoudi et al . (2022) 
 
Karim Hammoudi, Adnane Cabani, Bouthaina Slika, Halim Benhabiles, Fadi Dornaika, and Mahmoud Melkemi. 2022.

 
 SuperpixelGridCut, SuperpixelGridMean and SuperpixelGridMix data augmentation.

 
 (2022).

 
 arXiv:2204.08458

 

 
 Han et al . (2022) 
 
Xiaotian Han, Zhimeng Jiang, Ninghao Liu, and Xia Hu. 2022.

 
 G-Mixup: Graph data augmentation for graph classification. In International conference on machine learning . 8230–8248.

 
 
 

 
 Harris et al . (2020) 
 
Ethan Harris, Antonia Marcu, Matthew Painter, Mahesan Niranjan, Adam Prügel-Bennett, and Jonathon Hare. 2020.

 
 Fmix: Enhancing mixed sample data augmentation.

 
 (2020).

 
 arXiv:2002.12047

 

 
 Hataya and Nakayama (2022) 
 
Ryuichiro Hataya and Hideki Nakayama. 2022.

 
 DJMix: Unsupervised task-agnostic image augmentation for improving robustness of convolutional neural networks. In International joint conference on neural networks . 1–8.

 
 
 

 
 He et al . (2016) 
 
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016.

 
 Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition . 770–778.

 
 
 

 
 Hendrycks and Dietterich (2019) 
 
Dan Hendrycks and Thomas Dietterich. 2019.

 
 Benchmarking neural network robustness to common corruptions and perturbations. In International conference on learning representations .

 
 
 

 
 Hendrycks et al . (2020) 
 
Dan Hendrycks, Norman Mu, Ekin Dogus Cubuk, Barret Zoph, Justin Gilmer, and Balaji Lakshminarayanan. 2020.

 
 AugMix: A simple data processing method to improve robustness and uncertainty. In International conference on learning representations .

 
 
 

 
 Hendrycks et al . (2022) 
 
Dan Hendrycks, Andy Zou, Mantas Mazeika, Leonard Tang, Bo Li, Dawn Song, and Jacob Steinhardt. 2022.

 
 Pixmix: Dreamlike pictures comprehensively improve safety measures. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 16783–16792.

 
 
 

 
 Ho et al . (2020) 
 
Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020.

 
 Denoising diffusion probabilistic models. In Advances in neural information processing systems . 6840–6851.

 
 
 

 
 Hong et al . (2021) 
 
Minui Hong, Jinwoo Choi, and Gunhee Kim. 2021.

 
 Stylemix: Separating content and style for enhanced data augmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 14862–14870.

 
 
 

 
 Huang and Mumford (1999) 
 
Jinggang Huang and David Mumford. 1999.

 
 Statistics of natural images and models. In Proceedings IEEE computer society conference on computer vision and pattern recognition . 1541–1547.

 
 
 

 
 Huang et al . (2021) 
 
Shaoli Huang, Xinchao Wang, and Dacheng Tao. 2021.

 
 Snapmix: Semantically proportional mixing for augmenting fine-grained data. In Proceedings of the AAAI conference on artificial intelligence . 1628–1636.

 
 
 

 
 Huang and Belongie (2017) 
 
Xun Huang and Serge Belongie. 2017.

 
 Arbitrary style transfer in real-time with adaptive instance normalization. In Proceedings of the IEEE international conference on computer vision . 1510–1519.

 
 
 

 
 Hwang and Whang (2021) 
 
Seong-Hyeon Hwang and Steven Euijong Whang. 2021.

 
 MixRL: Data mixing augmentation for regression using reinforcement learning.

 
 (2021).

 
 arXiv:2106.03374

 

 
 Ioffe and Szegedy (2015) 
 
Sergey Ioffe and Christian Szegedy. 2015.

 
 Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning . 448–456.

 
 
 

 
 Jeong et al . (2021) 
 
Joonhyun Jeong, Sungmin Cha, Youngjoon Yoo, Sangdoo Yun, Taesup Moon, and Jongwon Choi. 2021.

 
 Observations on k-image expansion of image-mixing augmentation for classification.

 
 (2021).

 
 arXiv:2110.04248

 

 
 Jiao et al . (2023) 
 
Ruochen Jiao, Xiangguo Liu, Takami Sato, Qi Alfred Chen, and Qi Zhu. 2023.

 
 Semi-supervised semantics-guided adversarial training for robust trajectory prediction. In Proceedings of the IEEE/CVF international conference on computer vision . 8207–8217.

 
 
 

 
 Jindal et al . (2020) 
 
Amit Jindal, Arijit Ghosh Chowdhury, Aniket Didolkar, Di Jin, Ramit Sawhney, and Rajiv Shah. 2020.

 
 Augmenting NLP models using latent feature interpolations. In Proceedings of the international conference on computational linguistics . 6931–6936.

 
 
 

 
 Kalantidis et al . (2020) 
 
Yannis Kalantidis, Mert Bulent Sariyildiz, Noe Pion, Philippe Weinzaepfel, and Diane Larlus. 2020.

 
 Hard negative mixing for contrastive learning. In Advances in neural information processing systems . 21798–21809.

 
 
 

 
 Kenton and Toutanova (2019) 
 
Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. 2019.

 
 BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the conference of the North American chapter of the association for computational linguistics: human language technologies . 4171–4186.

 
 
 

 
 Kim et al . (2023) 
 
Daehan Kim, Minseok Seo, Kwanyong Park, Inkyu Shin, Sanghyun Woo, In So Kweon, and Dong-Geol Choi. 2023.

 
 Bidirectional Domain Mixup for Domain Adaptive Semantic Segmentation. In Proceedings of the AAAI conference on artificial intelligence . 1114–1123.

 
 
 

 
 Kim et al . (2021) 
 
JangHyun Kim, Wonho Choo, Hosan Jeong, and Hyun Oh Song. 2021.

 
 Co-Mixup: Saliency guided joint Mixup with supermodular diversity. In International conference on learning representations .

 
 
 

 
 Kim et al . (2020c) 
 
Jiyeon Kim, Ik-Hee Shin, Jong-Ryul Lee, and Yong-Ju Lee. 2020c.

 
 Where to cut and paste: Data regularization with selective features. In International conference on information and communication technology convergence . 1219–1221.

 
 
 

 
 Kim et al . (2020a) 
 
Jang-Hyun Kim, Wonho Choo, and Hyun Oh Song. 2020a.

 
 Puzzle mix: Exploiting saliency and local statistics for optimal Mixup. In International conference on machine learning . 5275–5285.

 
 
 

 
 Kim et al . (2020b) 
 
Sungnyun Kim, Gihun Lee, Sangmin Bae, and Se-Young Yun. 2020b.

 
 Mixco: Mixup contrastive learning for visual representation.

 
 (2020).

 
 arXiv:2010.06300

 

 
 Kingma and Welling (2014) 
 
Diederik P Kingma and Max Welling. 2014.

 
 Auto-encoding variational bayes. In International conference on learning representations .

 
 
 

 
 Ko and Gu (2020) 
 
Byungsoo Ko and Geonmo Gu. 2020.

 
 Embedding expansion: Augmentation in embedding space for deep metric learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 7253–7262.

 
 
 

 
 Kobayashi (2018) 
 
Sosuke Kobayashi. 2018.

 
 Contextual augmentation: Data augmentation by words with paradigmatic relations. In Proceedings of the conference of the North American chapter of the association for computational linguistics: human language technologies . 452–457.

 
 
 

 
 Kobyzev et al . (2020) 
 
Ivan Kobyzev, Simon JD Prince, and Marcus A Brubaker. 2020.

 
 Normalizing flows: An introduction and review of current methods.

 
 IEEE transactions on pattern analysis and machine intelligence 43, 11 (2020), 3964–3979.

 
 
 

 
 Kong et al . (2020) 
 
Lingkai Kong, Haoming Jiang, Yuchen Zhuang, Jie Lyu, Tuo Zhao, and Chao Zhang. 2020.

 
 Calibrated language model fine-tuning for in-and out-of-distribution data. In Proceedings of the conference on empirical methods in natural language processing . 1326–1340.

 
 
 

 
 Krizhevsky et al . (2009) 
 
Alex Krizhevsky, Geoffrey Hinton, et al . 2009.

 
 Learning multiple layers of features from tiny images .

 
 Technical Report. University of Toronto.

 
 
 

 
 Krizhevsky et al . (2012) 
 
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. 2012.

 
 Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems . 1106–1114.

 
 
 

 
 Kwon and Lee (2022) 
 
Soonki Kwon and Younghoon Lee. 2022.

 
 Explainability-based Mixup approach for text data augmentation.

 
 ACM transactions on knowledge discovery from data (2022).

 
 
 

 
 Lamb et al . (2019) 
 
Alex Lamb, Vikas Verma, Juho Kannala, and Yoshua Bengio. 2019.

 
 Interpolated adversarial training: Achieving robust neural networks without sacrificing too much accuracy. In Proceedings of the ACM workshop on artificial intelligence and security . 95–103.

 
 
 

 
 Larsen et al . (2016) 
 
Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, and Ole Winther. 2016.

 
 Autoencoding beyond pixels using a learned similarity metric. In International conference on machine learning . 1558–1566.

 
 
 

 
 Laugros et al . (2020) 
 
Alfred Laugros, Alice Caplier, and Matthieu Ospici. 2020.

 
 Addressing neural network robustness with Mixup and targeted labeling adversarial training. In European conference on computer vision . 178–195.

 
 
 

 
 LeCun et al . (2015) 
 
Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. 2015.

 
 Deep learning.

 
 Nature 521, 7553 (2015), 436–444.

 
 
 

 
 LeCun et al . (1998) 
 
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. 1998.

 
 Gradient-based learning applied to document recognition.

 
 Proc. IEEE 86, 11 (1998), 2278–2324.

 
 
 

 
 Lee et al . (2021a) 
 
Dogyoon Lee, Jaeha Lee, Junhyeop Lee, Hyeongmin Lee, Minhyeok Lee, Sungmin Woo, and Sangyoun Lee. 2021a.

 
 Regularization strategy for point cloud via rigidly mixed sample. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 15900–15909.

 
 
 

 
 Lee et al . (2020b) 
 
Jin-Ha Lee, Muhammad Zaigham Zaheer, Marcella Astrid, and Seung-Ik Lee. 2020b.

 
 Smoothmix: A simple yet effective data augmentation to train robust classifiers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 3264–3274.

 
 
 

 
 Lee et al . (2021b) 
 
Kibok Lee, Yian Zhu, Kihyuk Sohn, Chun-Liang Li, Jinwoo Shin, and Honglak Lee. 2021b.

 
 i i -Mix: A domain-agnostic strategy for contrastive representation learning. In International conference on learning representations .

 
 
 

 
 Lee et al . (2020a) 
 
Saehyung Lee, Hyungyu Lee, and Sungroh Yoon. 2020a.

 
 Adversarial vertex Mixup: Toward better adversarially robust generalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 272–281.

 
 
 

 
 Lewy and Mańdziuk (2022) 
 
Dominik Lewy and Jacek Mańdziuk. 2022.

 
 An overview of mixing augmentation methods and augmentation strategies.

 
 Artificial intelligence review (2022), 1–59.

 
 
 

 
 Li et al . (2022a) 
 
Bohan Li, Yutai Hou, and Wanxiang Che. 2022a.

 
 Data augmentation approaches in natural language processing: A survey.

 
 AI Open 3 (2022), 71–90.

 
 
 

 
 Li et al . (2021c) 
 
Boyi Li, Felix Wu, Ser-Nam Lim, Serge Belongie, and Kilian Q Weinberger. 2021c.

 
 On feature normalization and data augmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 12383–12392.

 
 
 

 
 Li et al . (2022b) 
 
Changchun Li, Ximing Li, Lei Feng, and Jihong Ouyang. 2022b.

 
 Who is your right Mixup partner in positive and unlabeled learning. In International conference on learning representations .

 
 
 

 
 Li et al . (2020c) 
 
Hao Li, Xiaopeng Zhang, Qi Tian, and Hongkai Xiong. 2020c.

 
 Attribute mix: Semantic data augmentation for fine grained recognition. In IEEE international conference on visual communications and image processing . 243–246.

 
 
 

 
 Li et al . (2020b) 
 
Hao Li, Xiaopeng Zhang, and Hongkai Xiong. 2020b.

 
 Center-wise local image mixture for contrastive representation learning. In Proceedings of the British machine vision conference . 369.

 
 
 

 
 Li et al . (2021a) 
 
Jicheng Li, Pengzhi Gao, Xuanfu Wu, Yang Feng, Zhongjun He, Hua Wu, and Haifeng Wang. 2021a.

 
 Mixup decoding for diverse machine translation. In Findings of the association for computational linguistics: EMNLP . 312–320.

 
 
 

 
 Li et al . (2020a) 
 
Junnan Li, Richard Socher, and Steven CH Hoi. 2020a.

 
 DivideMix: Learning with noisy labels as semi-supervised learning. In International conference on learning representations .

 
 
 

 
 Li et al . (2021b) 
 
Siyuan Li, Zicheng Liu, Di Wu, Zihan Liu, and Stan Z Li. 2021b.

 
 Boosting discriminative visual representation learning with scenario-agnostic Mixup.

 
 (2021).

 
 arXiv:2111.15454

 

 
 Liang et al . (2018) 
 
Daojun Liang, Feng Yang, Tian Zhang, and Peter Yang. 2018.

 
 Understanding Mixup training methods.

 
 IEEE access 6 (2018), 58774–58783.

 
 
 

 
 Lim et al . (2022) 
 
Soon Hoe Lim, N Benjamin Erichson, Francisco Utrera, Winnie Xu, and Michael W Mahoney. 2022.

 
 Noisy feature Mixup. In International conference on learning representations .

 
 
 

 
 Liu et al . (2021a) 
 
Guang Liu, Yuzhao Mao, Huang Hailong, Gao Weiguo, and Li Xuan. 2021a.

 
 Adversarial mixing policy for relaxing locally linear constraints in Mixup. In Proceedings of the conference on empirical methods in natural language processing . 2998–3008.

 
 
 

 
 Liu et al . (2022c) 
 
Jihao Liu, Boxiao Liu, Hang Zhou, Hongsheng Li, and Yu Liu. 2022c.

 
 TokenMix: Rethinking image mixing for data augmentation in vision transformers. In European conference on computer vision . 455–471.

 
 
 

 
 Liu et al . (2022d) 
 
Xiaoliang Liu, Furao Shen, Jian Zhao, and Changhai Nie. 2022d.

 
 AugRmixAT: A data processing and training method for improving multiple robustness and generalization performance. In IEEE international conference on multimedia and expo . 1–6.

 
 
 

 
 Liu et al . (2022e) 
 
Xiaoliang Liu, Furao Shen, Jian Zhao, and Changhai Nie. 2022e.

 
 RandomMix: A mixed sample data augmentation method with multiple mixed modes.

 
 (2022).

 
 arXiv:2205.08728

 

 
 Liu et al . (2021b) 
 
Xiao Liu, Fanjin Zhang, Zhenyu Hou, Li Mian, Zhaoyu Wang, Jing Zhang, and Jie Tang. 2021b.

 
 Self-supervised learning: Generative or contrastive.

 
 IEEE transactions on knowledge and data engineering (2021).

 
 
 

 
 Liu et al . (2018) 
 
Xiaofeng Liu, Yang Zou, Lingsheng Kong, Zhihui Diao, Junliang Yan, Jun Wang, Site Li, Ping Jia, and Jane You. 2018.

 
 Data augmentation via latent space interpolation for image classification. In International conference on pattern recognition . 728–733.

 
 
 

 
 Liu et al . (2022a) 
 
Zicheng Liu, Siyuan Li, Ge Wang, Cheng Tan, Lirong Wu, and Stan Z Li. 2022a.

 
 Decoupled Mixup for data-efficient learning.

 
 (2022).

 
 arXiv:2203.10761

 

 
 Liu et al . (2022b) 
 
Zicheng Liu, Siyuan Li, Di Wu, Zhiyuan Chen, Lirong Wu, Jianzhu Guo, and Stan Z Li. 2022b.

 
 Unveiling the power of Mixup for stronger classifiers. In European conference on computer vision . 441–458.

 
 
 

 
 Ma et al . (2023) 
 
Ning Ma, Haishuai Wang, Zhen Zhang, Sheng Zhou, Hongyang Chen, and Jiajun Bu. 2023.

 
 Source-free semi-supervised domain adaptation via progressive Mixup.

 
 Knowledge-based systems 262 (2023), 110208.

 
 
 

 
 Ma et al . (2022) 
 
Qianli Ma, Zheng Fan, Chenzhi Wang, and Hongye Tan. 2022.

 
 Graph mixed random network based on pageRank.

 
 Symmetry 14, 8 (2022), 1678.

 
 
 

 
 Maas et al . (2011) 
 
Andrew Maas, Raymond E Daly, Peter T Pham, Dan Huang, Andrew Y Ng, and Christopher Potts. 2011.

 
 Learning word vectors for sentiment analysis. In Proceedings of the annual meeting of the association for computational linguistics: human language technologies . 142–150.

 
 
 

 
 Madry et al . (2018) 
 
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. 2018.

 
 Towards deep learning models resistant to adversarial attacks. In International conference on learning representations .

 
 
 

 
 Mai et al . (2021) 
 
Zhijun Mai, Guosheng Hu, Dexiong Chen, Fumin Shen, and Heng Tao Shen. 2021.

 
 MetaMixup: Learning adaptive interpolation policy of Mixup with metalearning.

 
 IEEE transactions on neural networks and learning systems 33, 7 (2021), 3050–3064.

 
 
 

 
 Maji et al . (2013) 
 
Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. 2013.

 
 Fine-grained visual classification of aircraft.

 
 (2013).

 
 arXiv:1306.5151

 

 
 Mangla et al . (2020) 
 
Puneet Mangla, Vedant Singh, Shreyas Jayant Havaldar, and Vineeth N Balasubramanian. 2020.

 
 VarMixup: Exploiting the latent space for robust training and inference.

 
 (2020).

 
 arXiv:2003.06566

 

 
 Mao et al . (2019) 
 
Xudong Mao, Yun Ma, Zhenguo Yang, Yangbin Chen, and Qing Li. 2019.

 
 Virtual Mixup training for unsupervised domain adaptation.

 
 (2019).

 
 arXiv:1905.04215

 

 
 Meng et al . (2021) 
 
Linghui Meng, Jin Xu, Xu Tan, Jindong Wang, Tao Qin, and Bo Xu. 2021.

 
 MixSpeech: Data augmentation for low-resource automatic speech recognition. In IEEE international conference on acoustics, speech and signal processing . 7008–7012.

 
 
 

 
 Miao et al . (2020) 
 
Zhengjie Miao, Yuliang Li, Xiaolan Wang, and Wang-Chiew Tan. 2020.

 
 Snippext: Semi-supervised opinion mining with augmented data. In Proceedings of the web conference . 617–628.

 
 
 

 
 Mikolov et al . (2013) 
 
Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013.

 
 Efficient estimation of word representations in vector space. In International conference on learning representations .

 
 
 

 
 Montabone and Soto (2010) 
 
Sebastian Montabone and Alvaro Soto. 2010.

 
 Human detection using a mobile platform and novel features derived from a visual saliency mechanism.

 
 Image and vision computing 28, 3 (2010), 391–402.

 
 
 

 
 Moon et al . (2023) 
 
Jaewan Moon, Yoonki Jeong, Dong-Kyu Chae, Jaeho Choi, Hyunjung Shim, and Jongwuk Lee. 2023.

 
 CoMix: Collaborative filtering with mixup for implicit datasets.

 
 Information sciences 628 (2023), 254–268.

 
 
 

 
 Muhammad et al . (2021) 
 
Awais Muhammad, Fengwei Zhou, Chuanlong Xie, Jiawei Li, Sung-Ho Bae, and Zhenguo Li. 2021.

 
 MixACM: Mixup-based robustness transfer via distillation of activated channel maps. In Advances in neural information processing systems . 4555–4569.

 
 
 

 
 Naveed et al . (2024) 
 
Humza Naveed, Saeed Anwar, Munawar Hayat, Kashif Javed, and Ajmal Mian. 2024.

 
 Survey: Image mixing and deleting for data augmentation.

 
 Engineering applications of artificial intelligence 131 (2024), 107791.

 
 
 

 
 Netzer et al . (2011) 
 
Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. 2011.

 
 Reading digits in natural images with unsupervised feature learning. In Advances in neural information processing systems .

 
 
 

 
 Olsson et al . (2021) 
 
Viktor Olsson, Wilhelm Tranheden, Juliano Pinto, and Lennart Svensson. 2021.

 
 Classmix: Segmentation-based data augmentation for semi-supervised learning. In Proceedings of the IEEE/CVF winter conference on applications of computer vision . 1368–1377.

 
 
 

 
 Pang and Lee (2004) 
 
Bo Pang and Lillian Lee. 2004.

 
 A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In Proceedings of the annual meeting on association for computational linguistics . 271–278.

 
 
 

 
 Pang and Lee (2005) 
 
Bo Pang and Lillian Lee. 2005.

 
 Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the annual meeting of the association for computational linguistics . 115–124.

 
 
 

 
 Pang et al . (2020) 
 
Tianyu Pang, Kun Xu, and Jun Zhu. 2020.

 
 Mixup inference: Better exploiting Mixup to defend adversarial attacks. In International conference on learning representations .

 
 
 

 
 Park et al . (2022c) 
 
Chanwoo Park, Sangdoo Yun, and Sanghyuk Chun. 2022c.

 
 A unified analysis of mixed sample data augmentation: A loss function perspective. In Advances in neural information processing systems . 35504–35518.

 
 
 

 
 Park et al . (2022a) 
 
Joonhyung Park, Hajin Shim, and Eunho Yang. 2022a.

 
 Graph transplant: Node saliency-guided graph Mixup with local structure preservation. In Proceedings of the AAAI conference on artificial intelligence . 7966–7974.

 
 
 

 
 Park et al . (2022b) 
 
Joonhyung Park, June Yong Yang, Jinwoo Shin, Sung Ju Hwang, and Eunho Yang. 2022b.

 
 Saliency grafting: Innocuous attribution-guided Mixup with calibrated label mixing. In Proceedings of the AAAI conference on artificial intelligence . 7957–7965.

 
 
 

 
 Park and Caragea (2022) 
 
Seo Yeon Park and Cornelia Caragea. 2022.

 
 On the calibration of pre-trained language models using Mixup guided by area under the margin and saliency. In Proceedings of the annual meeting of the association for computational linguistics . 5364–5374.

 
 
 

 
 Patel et al . (2022) 
 
Yash Patel, Giorgos Tolias, and Jiří Matas. 2022.

 
 Recall@k surrogate loss with large batches and similarity Mixup. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 7492–7501.

 
 
 

 
 Pereira and dos Santos (2021) 
 
Matheus Barros Pereira and Jefersson Alex dos Santos. 2021.

 
 ChessMix: Spatial context data augmentation for remote sensing semantic segmentation. In SIBGRAPI conference on graphics, patterns and images . 278–285.

 
 
 

 
 Pereyra et al . (2017) 
 
Gabriel Pereyra, George Tucker, Jan Chorowski, Łukasz Kaiser, and Geoffrey Hinton. 2017.

 
 Regularizing neural networks by penalizing confident output distributions.

 
 (2017).

 
 arXiv:1701.06548

 

 
 Pinto et al . (2022) 
 
Francesco Pinto, Harry Yang, Ser-Nam Lim, Philip HS Torr, and Puneet K Dokania. 2022.

 
 RegMixup: Mixup as a regularizer can surprisingly improve accuracy and out distribution robustness.

 
 (2022).

 
 arXiv:2206.14502

 

 
 Qin et al . (2020) 
 
Jie Qin, Jiemin Fang, Qian Zhang, Wenyu Liu, Xingang Wang, and Xinggang Wang. 2020.

 
 Resizemix: Mixing data with preserved object information and true labels.

 
 (2020).

 
 arXiv:2012.11101

 

 
 Queiroz Abonizio and Barbon Junior (2020) 
 
Hugo Queiroz Abonizio and Sylvio Barbon Junior. 2020.

 
 Pre-trained data augmentation for text classification. In Brazilian Conference on Intelligent Systems . 551–565.

 
 
 

 
 Ramé et al . (2021) 
 
Alexandre Ramé, Rémy Sun, and Matthieu Cord. 2021.

 
 Mixmo: Mixing multiple inputs for multiple outputs via deep subnetworks. In Proceedings of the IEEE/CVF international conference on computer vision . 803–813.

 
 
 

 
 Ren et al . (2022) 
 
Sucheng Ren, Huiyu Wang, Zhengqi Gao, Shengfeng He, Alan Yuille, Yuyin Zhou, and Cihang Xie. 2022.

 
 A Simple data mixing Prior for improving self-supervised learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 14575–14584.

 
 
 

 
 Russakovsky et al . (2015) 
 
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al . 2015.

 
 Imagenet large scale visual recognition challenge.

 
 International journal of computer vision 115, 3 (2015), 211–252.

 
 
 

 
 Sahoo et al . (2021a) 
 
Aadarsh Sahoo, Rameswar Panda, Rogerio Feris, Kate Saenko, and Abir Das. 2021a.

 
 Select, label, and mix: Learning discriminative invariant feature representations for partial domain adaptation. In NeurIPS workshop on distribution shifts: Connecting methods and applications .

 
 
 

 
 Sahoo et al . (2021b) 
 
Aadarsh Sahoo, Rutav Shah, Rameswar Panda, Kate Saenko, and Abir Das. 2021b.

 
 Contrast and mix: Temporal contrastive video domain adaptation with background mixing. In Advances in neural information processing systems . 23386–23400.

 
 
 

 
 Sawhney et al . (2021) 
 
Ramit Sawhney, Megh Thakkar, Shivam Agarwal, Di Jin, Diyi Yang, and Lucie Flek. 2021.

 
 HYPMIX: Hyperbolic interpolative data augmentation. In Proceedings of the conference on empirical methods in natural language processing .

 
 
 

 
 Sawhney et al . (2022) 
 
Ramit Sawhney, Megh Thakkar, Shrey Pandit, Ritesh Soun, Di Jin, Diyi Yang, and Lucie Flek. 2022.

 
 DMIX: Adaptive distance-aware interpolative Mixup. In Proceedings of the annual meeting of the association for computational linguistics . 606–612.

 
 
 

 
 Schmidt et al . (2018) 
 
Ludwig Schmidt, Shibani Santurkar, Dimitris Tsipras, Kunal Talwar, and Aleksander Madry. 2018.

 
 Adversarially robust generalization requires more data. In Advances in neural information processing systems . 5019–5031.

 
 
 

 
 Shen et al . (2020) 
 
Dinghan Shen, Mingzhi Zheng, Yelong Shen, Yanru Qu, and Weizhu Chen. 2020.

 
 A simple but tough-to-beat data augmentation approach for natural language understanding and generation.

 
 (2020).

 
 arXiv:2009.13818

 

 
 Shen et al . (2022) 
 
Zhiqiang Shen, Zechun Liu, Zhuang Liu, Marios Savvides, Trevor Darrell, and Eric Xing. 2022.

 
 Un-mix: Rethinking image mixtures for unsupervised visual representation learning. In Proceedings of the AAAI conference on artificial intelligence . 2216–2224.

 
 
 

 
 Shorten and Khoshgoftaar (2019) 
 
Connor Shorten and Taghi M Khoshgoftaar. 2019.

 
 A survey on image data augmentation for deep learning.

 
 Journal of big data 6, 1 (2019), 1–48.

 
 
 

 
 Si et al . (2021) 
 
Chenglei Si, Zhengyan Zhang, Fanchao Qi, Zhiyuan Liu, Yasheng Wang, Qun Liu, and Maosong Sun. 2021.

 
 Better robustness by more coverage: Adversarial and Mixup data augmentation for robust finetuning. In Findings of the association for computational linguistics: ACL/IJCNLP . 1569–1576.

 
 
 

 
 Socher et al . (2013) 
 
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. 2013.

 
 Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the conference on empirical methods in natural language processing . 1631–1642.

 
 
 

 
 Sohn et al . (2022) 
 
Jy-yong Sohn, Liang Shang, Hongxu Chen, Jaekyun Moon, Dimitris Papailiopoulos, and Kangwook Lee. 2022.

 
 GenLabel: Mixup relabeling using generative models. In International conference on machine learning . 20278–20313.

 
 
 

 
 Srivastava et al . (2014) 
 
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014.

 
 Dropout: A simple way to prevent neural networks from overfitting.

 
 The journal of machine learning research 15, 1 (2014), 1929–1958.

 
 
 

 
 Stegmüller et al . (2023) 
 
Thomas Stegmüller, Behzad Bozorgtabar, Antoine Spahr, and Jean-Philippe Thiran. 2023.

 
 Scorenet: Learning non-uniform attention and augmentation for transformer-based histopathological image classification. In Proceedings of the IEEE/CVF winter conference on applications of computer vision . 6170–6179.

 
 
 

 
 Summers and Dinneen (2019) 
 
Cecilia Summers and Michael J Dinneen. 2019.

 
 Improved mixed-example data augmentation. In Proceedings of the IEEE/CVF winter conference on applications of computer vision . 1262–1270.

 
 
 

 
 Sun et al . (2024) 
 
Ke Sun, Bing Yu, Zhouchen Lin, and Zhanxing Zhu. 2024.

 
 Patch-level neighborhood interpolation: A general and effective graph-based regularization strategy. In Asian conference on machine learning . 1276–1291.

 
 
 

 
 Sun et al . (2020) 
 
Lichao Sun, Congying Xia, Wenpeng Yin, Tingting Liang, S Yu Philip, and Lifang He. 2020.

 
 Mixup-transformer: Dynamic data augmentation for NLP tasks. In Proceedings of the international conference on computational linguistics . 3436–3440.

 
 
 

 
 Sun et al . (2022) 
 
Rémy Sun, Clément Masson, Gilles Hénaff, Nicolas Thome, and Matthieu Cord. 2022.

 
 Swapping semantic contents for mixing images. In International conference on learning representations .

 
 
 

 
 Szegedy et al . (2016) 
 
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. 2016.

 
 Rethinking the inception architecture for computer vision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 2818–2826.

 
 
 

 
 Szegedy et al . (2014) 
 
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. 2014.

 
 Intriguing properties of neural networks. In International conference on learning representations .

 
 
 

 
 Takahashi et al . (2018) 
 
Ryo Takahashi, Takashi Matsubara, and Kuniaki Uehara. 2018.

 
 Ricap: Random image cropping and patching data augmentation for deep cnns. In Asian conference on machine learning . 786–798.

 
 
 

 
 Thulasidasan et al . (2019) 
 
Sunil Thulasidasan, Gopinath Chennupati, Jeff A Bilmes, Tanmoy Bhattacharya, and Sarah Michalak. 2019.

 
 On Mixup training: Improved calibration and predictive uncertainty for deep neural networks. In Advances in neural information processing systems . 13888–13899.

 
 
 

 
 Tokozume et al . (2018a) 
 
Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada. 2018a.

 
 Between-class learning for image classification. In Proceedings of the IEEE conference on computer vision and pattern recognition . 5486–5494.

 
 
 

 
 Tokozume et al . (2018b) 
 
Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada. 2018b.

 
 Learning from between-class examples for deep sound recognition. In International conference on learning representations .

 
 
 

 
 Uddin et al . (2021) 
 
AFM Shahab Uddin, Mst Sirazam Monira, Wheemyung Shin, TaeChoong Chung, and Sung-Ho Bae. 2021.

 
 SaliencyMix: A saliency guided data augmentation strategy for better regularization. In International conference on learning representations .

 
 
 

 
 Van Horn et al . (2018) 
 
Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. 2018.

 
 The inaturalist species classification and detection dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition . 8769–8778.

 
 
 

 
 Vaswani et al . (2017) 
 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017.

 
 Attention is all you need. In Advances in neural information processing systems . 5998–6008.

 
 
 

 
 Venkataramanan et al . (2022a) 
 
Shashanka Venkataramanan, Ewa Kijak, Laurent Amsaleg, and Yannis Avrithis. 2022a.

 
 AlignMixup: Improving representations by interpolating aligned features. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 19174–19183.

 
 
 

 
 Venkataramanan et al . (2022b) 
 
Shashanka Venkataramanan, Ewa Kijak, Laurent Amsaleg, and Yannis Avrithis. 2022b.

 
 Teach me how to interpolate a myriad of embeddings.

 
 (2022).

 
 arXiv:2206.14868

 

 
 Venkataramanan et al . (2022c) 
 
Shashanka Venkataramanan, Bill Psomas, Ewa Kijak, Konstantinos Karantzalos, Yannis Avrithis, et al . 2022c.

 
 It takes two to tango: Mixup for deep metric learning. In International conference on learning representations .

 
 
 

 
 Verma et al . (2019a) 
 
Vikas Verma, Alex Lamb, Christopher Beckham, Amir Najafi, Ioannis Mitliagkas, David Lopez-Paz, and Yoshua Bengio. 2019a.

 
 Manifold Mixup: Better representations by interpolating hidden states. In International conference on machine learning . 6438–6447.

 
 
 

 
 Verma et al . (2019b) 
 
Vikas Verma, Alex Lamb, Juho Kannala, Yoshua Bengio, and David Lopez-Paz. 2019b.

 
 Interpolation consistency training for semi-supervised learning. In Proceedings of the international joint conference on artificial intelligence . 3635–3641.

 
 
 

 
 Verma et al . (2021) 
 
Vikas Verma, Meng Qu, Kenji Kawaguchi, Alex Lamb, Yoshua Bengio, Juho Kannala, and Jian Tang. 2021.

 
 Graphmix: Improved training of GNNs for semi-supervised learning. In Proceedings of the AAAI conference on artificial intelligence . 10024–10032.

 
 
 

 
 Wager et al . (2013) 
 
Stefan Wager, Sida Wang, and Percy S Liang. 2013.

 
 Dropout training as adaptive regularization. In Advances in Neural Information Processing Systems . 351–359.

 
 
 

 
 Wah et al . (2011) 
 
Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. 2011.

 
 The caltech-ucsd birds-200-2011 dataset .

 
 
 

 
 Walawalkar et al . (2020) 
 
Devesh Walawalkar, Zhiqiang Shen, Zechun Liu, and Marios Savvides. 2020.

 
 Attentive Cutmix: An enhanced data augmentation approach for deep learning based image classification. In IEEE international conference on acoustics, speech and signal processing . 3642–3646.

 
 
 

 
 Wang et al . (2019) 
 
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2019.

 
 GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International conference on learning representations .

 
 
 

 
 Wang et al . (2023) 
 
Deng-Bao Wang, Lanqing Li, Peilin Zhao, Pheng-Ann Heng, and Min-Ling Zhang. 2023.

 
 On the pitfall of Mixup for uncertainty calibration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 7609–7618.

 
 
 

 
 Wang et al . (2022) 
 
Teng Wang, Wenhao Jiang, Zhichao Lu, Feng Zheng, Ran Cheng, Chengguo Yin, and Ping Luo. 2022.

 
 VLMixer: Unpaired vision-language Pre-training via cross-modal CutMix. In International conference on machine learning . 22680–22690.

 
 
 

 
 Wang et al . (2021) 
 
Yiwei Wang, Wei Wang, Yuxuan Liang, Yujun Cai, and Bryan Hooi. 2021.

 
 Mixup for node and graph classification. In Proceedings of the web conference . 3663–3674.

 
 
 

 
 Warden (2018) 
 
Pete Warden. 2018.

 
 Speech commands: A dataset for limited-vocabulary speech recognition.

 
 (2018).

 
 arXiv:1804.03209

 

 
 Wei et al . (2020a) 
 
Colin Wei, Sham Kakade, and Tengyu Ma. 2020a.

 
 The implicit and explicit regularization effects of dropout. In International Conference on Machine Learning . 10181–10192.

 
 
 

 
 Wei and Zou (2019) 
 
Jason Wei and Kai Zou. 2019.

 
 EDA: Easy data augmentation techniques for boosting performance on text classification tasks. In Proceedings of the conference on empirical methods in natural language processing and the international joint conference on natural language processing . 6382–6388.

 
 
 

 
 Wei et al . (2020b) 
 
Tong Wei, Feng Shi, Hai Wang, Wei-Wei Tu Li, et al . 2020b.

 
 MixPUL: Consistency-based augmentation for positive and unlabeled learning.

 
 (2020).

 
 arXiv:2004.09388

 

 
 Wei et al . (2022) 
 
Xiangpeng Wei, Heng Yu, Yue Hu, Rongxiang Weng, Weihua Luo, and Rong Jin. 2022.

 
 Learning to generalize to more: Continuous semantic augmentation for neural machine translation. In Proceedings of the annual meeting of the association for computational linguistics . 7930–7944.

 
 
 

 
 Wen et al . (2021b) 
 
Qingsong Wen, Liang Sun, Fan Yang, Xiaomin Song, Jingkun Gao, Xue Wang, and Huan Xu. 2021b.

 
 Time series data augmentation for deep learning: A survey. In Proceedings of the international joint conference on artificial intelligence . 4653–4660.

 
 
 

 
 Wen et al . (2021a) 
 
Yeming Wen, Ghassen Jerfel, Rafael Muller, Michael W Dusenberry, Jasper Snoek, Balaji Lakshminarayanan, and Dustin Tran. 2021a.

 
 Combining ensembles and data augmentation can harm your calibration. In International conference on learning representations .

 
 
 

 
 Wickstrøm et al . (2022) 
 
Kristoffer Wickstrøm, Michael Kampffmeyer, Karl Øyvind Mikalsen, and Robert Jenssen. 2022.

 
 Mixing up contrastive learning: Self-supervised representation learning for time series.

 
 Pattern recognition letters 155 (2022), 54–61.

 
 
 

 
 Wu et al . (2024) 
 
Chenwang Wu, Defu Lian, Yong Ge, Min Zhou, Enhong Chen, and Dacheng Tao. 2024.

 
 Boosting factorization machines via saliency-guided Mixup.

 
 IEEE transactions on pattern analysis and machine intelligence 46, 6 (2024), 4443–4459.

 
 
 

 
 Wu et al . (2021) 
 
Lirong Wu, Haitao Lin, Zhangyang Gao, Cheng Tan, Stan Li, et al . 2021.

 
 GraphMixup: Improving class-imbalanced node classification on graphs by self-supervised context prediction.

 
 (2021).

 
 arXiv:2106.11133

 

 
 Wu et al . (2020) 
 
Yuan Wu, Diana Inkpen, and Ahmed El-Roby. 2020.

 
 Dual Mixup regularized learning for adversarial domain adaptation. In European conference on computer vision . 540–555.

 
 
 

 
 Xia et al . (2022) 
 
Jun Xia, Lirong Wu, Ge Wang, Jintao Chen, and Stan Z Li. 2022.

 
 ProGCL: Rethinking hard negative mining in graph contrastive learning. In International conference on machine learning . 24332–24346.

 
 
 

 
 Xiao et al . (2023) 
 
Han Xiao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, and Jiwen Lu. 2023.

 
 Token-label alignment for vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision . 5495–5504.

 
 
 

 
 Xu et al . (2023) 
 
Guodong Xu, Ziwei Liu, and Chen Change Loy. 2023.

 
 Computation-efficient knowledge distillation via uncertainty-aware Mixup.

 
 Pattern recognition 138 (2023), 109338.

 
 
 

 
 Xu et al . (2020) 
 
Minghao Xu, Jian Zhang, Bingbing Ni, Teng Li, Chengjie Wang, Qi Tian, and Wenjun Zhang. 2020.

 
 Adversarial domain adaptation with domain Mixup. In Proceedings of the AAAI conference on artificial intelligence . 6502–6509.

 
 
 

 
 Xue et al . (2021) 
 
Yifan Xue, Yixuan Liao, Xiaoxin Chen, and Jingwei Zhao. 2021.

 
 Node augmentation methods for graph neural network based object classification. In International conference on computing and data science . 556–561.

 
 
 

 
 Yan et al . (2020) 
 
Shen Yan, Huan Song, Nanxiang Li, Lincan Zou, and Liu Ren. 2020.

 
 Improve unsupervised domain adaptation with Mixup training.

 
 (2020).

 
 arXiv:2001.00677

 

 
 Yang et al . (2022a) 
 
Huiyun Yang, Huadong Chen, Hao Zhou, and Lei Li. 2022a.

 
 Enhancing cross-lingual transfer by manifold Mixup. In International conference on learning representations .

 
 
 

 
 Yang et al . (2022b) 
 
Lingfeng Yang, Xiang Li, Borui Zhao, Renjie Song, and Jian Yang. 2022b.

 
 Recursivemix: Mixed learning with history. In Advances in neural information processing systems . 8427–8440.

 
 
 

 
 Yang et al . (2022c) 
 
Suorong Yang, Weikang Xiao, Mengcheng Zhang, Suhan Guo, Jian Zhao, and Furao Shen. 2022c.

 
 Image data augmentation for deep learning: A survey.

 
 (2022).

 
 arXiv:2204.08610

 

 
 Yao et al . (2021) 
 
Huaxiu Yao, Long-Kai Huang, Linjun Zhang, Ying Wei, Li Tian, James Zou, Junzhou Huang, et al . 2021.

 
 Improving generalization in meta-learning via task augmentation. In International conference on machine learning . 11887–11897.

 
 
 

 
 Yin et al . (2021) 
 
Wenpeng Yin, Huan Wang, Jin Qu, and Caiming Xiong. 2021.

 
 BatchMixup: Improving training by interpolating hidden states of the entire mini-batch. In Findings of the association for computational linguistics: ACL/IJCNLP . 4908–4912.

 
 
 

 
 Yoo et al . (2020) 
 
Jaejun Yoo, Namhyuk Ahn, and Kyung-Ah Sohn. 2020.

 
 Rethinking data augmentation for image super-resolution: A comprehensive analysis and a new strategy. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 8372–8381.

 
 
 

 
 Yoo et al . (2021) 
 
Kang Min Yoo, Dongju Park, Jaewook Kang, Sang-Woo Lee, and Woomyoung Park. 2021.

 
 GPT3Mix: leveraging large-scale language models for text augmentation. In Findings of the association for computational linguistics: EMNLP . 2225–2239.

 
 
 

 
 Yoon et al . (2021a) 
 
Soyoung Yoon, Gyuwan Kim, and Kyumin Park. 2021a.

 
 SSMix: Saliency-based span Mixup for text classification. In Findings of the association for computational linguistics: ACL/IJCNLP . 3225–3234.

 
 
 

 
 Yoon et al . (2021b) 
 
Tehrim Yoon, Sumin Shin, Sung Ju Hwang, and Eunho Yang. 2021b.

 
 FedMix: Approximation of Mixup under mean augmented federated learning. In International conference on learning representations .

 
 
 

 
 Yu et al . (2021) 
 
Hao Yu, Huanyu Wang, and Jianxin Wu. 2021.

 
 Mixup without hesitation. In International conference on image and graphics . 143–154.

 
 
 

 
 Yun et al . (2019) 
 
Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. 2019.

 
 Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF international conference on computer vision . 6023–6032.

 
 
 

 
 Yun et al . (2020) 
 
Sangdoo Yun, Seong Joon Oh, Byeongho Heo, Dongyoon Han, and Jinhyung Kim. 2020.

 
 Videomix: Rethinking data augmentation for video classification.

 
 (2020).

 
 arXiv:2012.03457

 

 
 Zhang et al . (2018) 
 
Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. 2018.

 
 Mixup: Beyond empirical risk minimization. In International conference on learning representations .

 
 
 

 
 Zhang et al . (2020a) 
 
Jize Zhang, Bhavya Kailkhura, and T Yong-Jin Han. 2020a.

 
 Mix-n-match: Ensemble and compositional methods for uncertainty calibration in deep learning. In International Conference on Machine Learning . 11117–11128.

 
 
 

 
 Zhang et al . (2021a) 
 
Linjun Zhang, Zhun Deng, Kenji Kawaguchi, Amirata Ghorbani, and James Zou. 2021a.

 
 How does Mixup help with robustness and generalization?. In International conference on learning representations .

 
 
 

 
 Zhang et al . (2022a) 
 
Linjun Zhang, Zhun Deng, Kenji Kawaguchi, and James Zou. 2022a.

 
 When and how Mixup improves calibration. In International conference on machine learning . 26135–26160.

 
 
 

 
 Zhang et al . (2022d) 
 
Le Zhang, Zichao Yang, and Diyi Yang. 2022d.

 
 TreeMix: Compositional constituency-based data augmentation for natural language understanding. In Proceedings of the conference of the North American chapter of the association for computational linguistics: human language technologies . 5243–5258.

 
 
 

 
 Zhang et al . (2020b) 
 
Rongzhi Zhang, Yue Yu, and Chao Zhang. 2020b.

 
 SeqMix: Augmenting active sequence labeling via sequence Mixup. In Proceedings of the conference on empirical methods in natural language processing . 8566–8579.

 
 
 

 
 Zhang et al . (2022c) 
 
Shaofeng Zhang, Meng Liu, Junchi Yan, Hengrui Zhang, Lingxiao Huang, Xiaokang Yang, and Pinyan Lu. 2022c.

 
 M-Mix: Generating hard negatives via multi-sample mixing for contrastive learning. In Proceedings of the ACM SIGKDD conference on knowledge discovery and data mining . 2461–2470.

 
 
 

 
 Zhang et al . (2022b) 
 
Xin Zhang, Minho Jin, Roger Cheng, Ruirui Li, Eunjung Han, and Andreas Stolcke. 2022b.

 
 Contrastive-Mixup learning for improved speaker verification. In IEEE international conference on acoustics, speech and signal processing . 7652–7656.

 
 
 

 
 Zhang et al . (2021b) 
 
Yifan Zhang, Bryan Hooi, Dapeng Hu, Jian Liang, and Jiashi Feng. 2021b.

 
 Unleashing the power of contrastive self-supervised visual models via contrast-regularized fine-tuning. In Advances in neural information processing systems . 29848–29860.

 
 
 

 
 Zhao and Lei (2021) 
 
Caidan Zhao and Yang Lei. 2021.

 
 Intra-class Cutmix for unbalanced data augmentation. In International conference on machine learning and computing . 246–251.

 
 
 

 
 Zhao et al . (2021) 
 
Tong Zhao, Yozen Liu, Leonardo Neves, Oliver Woodford, Meng Jiang, and Neil Shah. 2021.

 
 Data augmentation for graph neural networks. In Proceedings of the AAAI conference on artificial intelligence . 11015–11023.

 
 
 

 
 Zhao et al . (2022) 
 
Tianxiang Zhao, Xiang Zhang, and Suhang Wang. 2022.

 
 Synthetic over-sampling for imbalanced node classification with graph neural networks.

 
 (2022).

 
 arXiv:2206.05335

 

 
 Zheng et al . (2019) 
 
Wenzhao Zheng, Zhaodong Chen, Jiwen Lu, and Jie Zhou. 2019.

 
 Hardness-aware deep metric learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 72–81.

 
 
 

 
 Zhong et al . (2020) 
 
Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. 2020.

 
 Random erasing data augmentation. In Proceedings of the AAAI conference on artificial intelligence . 13001–13008.

 
 
 

 
 Zhou et al . (2021) 
 
Kaiyang Zhou, Yongxin Yang, Yu Qiao, and Tao Xiang. 2021.

 
 Domain generalization with MixStyle. In International conference on learning representations .

 
 
 

 
 Zhu et al . (2020) 
 
Jianchao Zhu, Liangliang Shi, Junchi Yan, and Hongyuan Zha. 2020.

 
 Automix: Mixup networks for sample interpolation via cooperative barycenter learning. In European conference on computer vision . 633–649.

 
 
 

 
 Zhu et al . (2021) 
 
Rui Zhu, Bingchen Zhao, Jingen Liu, Zhenglong Sun, and Chang Wen Chen. 2021.

 
 Improving contrastive learning by visualizing feature transformation. In Proceedings of the IEEE/CVF international conference on computer vision . 10286–10295.