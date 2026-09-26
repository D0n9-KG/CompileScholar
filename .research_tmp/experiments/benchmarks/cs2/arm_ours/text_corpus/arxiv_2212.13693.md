Parsing Objects at a Finer Granularity: A Survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2212.13693v1 [cs.CV] 28 Dec 2022 
 
 
 
 ﻿

 

# 
Parsing Objects at a Finer Granularity: A Survey

 
 
 Yifan Zhao
 
    
 Jia Li
 
    
 and Yonghong Tian
 † † thanks: 
Y. Zhao and Y. Tian are with the School of Computer Science, Peking University, Beijing, 100871, China
J. Li is with the State Key Laboratory of Virtual Reality Technology and Systems, School of Computer Science and Engineering, Beihang University, Beijing, 100191, China.
Y. Tian is also with School of Electronic and Computer Engineering, Peking University Shenzhen Graduate School, Peking University, Shenzhen, 518055, China.
J. Li is the corresponding author (E-mail: jiali@buaa.edu.cn).
 

 Abstract 
 
 Fine-grained visual parsing, including fine-grained part segmentation and fine-grained object recognition, has attracted considerable critical attention due to its importance in many real-world applications,  e.g. , agriculture, remote sensing, and space technologies. Predominant research efforts tackle these fine-grained sub-tasks following different paradigms, while the inherent relations between these tasks are neglected. Moreover, given most of the research remains fragmented, we conduct an in-depth study of the advanced work from a new perspective of learning the part relationship. In this perspective, we first consolidate recent research and benchmark syntheses with new taxonomies. Based on this consolidation, we revisit the universal challenges in fine-grained part segmentation and recognition tasks and propose new solutions by part relationship learning for these important challenges. Furthermore, we conclude several promising lines of research in fine-grained visual parsing for future research.

 
 
 
 Index Terms:  Fine-grained, visual parsing, part segmentation, fine-grained object recognition, part relationship

 
 

## I Introduction 

 
 Fig. 1: Comparisons of coarse-grained learning and fine-grained learning. Representative fine-grained learning tasks,  i.e. , semantic part segmentation and fine-grained recognition, rely on the part relationship learning to build robust local features, while the coarse-grained tasks can be achieved by image-level global features.
 
 
 
 Fine-grained visual parsing of image objects is a basic and crucial task in the computer vision community, which is fundamentally difficult, owing that there are usually subtle visual cues for distinguishing different objects or part regions. Recent advances in deep learning have significantly boosted the image understanding abilities of machine systems,  e.g. , its performance on the large-scale ImageNet dataset  [ 1 ] surpasses the human-level recognition, but it is still a great challenge facing the fine-grained visual tasks. In particular, we consider two representative fine-grained visual parsing tasks in this paper,  i.e. , semantic part segmentation and fine-grained object recognition.

 
 
 In contrast with coarse-grained object segmentation and base-level classification, fine-grained parsing is meant to segment or distinguish visually similar objects that belong to different fine-grained concepts, for example, decomposing objects into parts and dividing the base category into subcategories.
A tremendous amount of research efforts  [ 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 ] has been proposed to solve this important problem, which can also be applied for downstream applications  [ 11 , 12 , 13 ] . Conventional machine learning techniques build explicit structures for parsing and understanding these fine-grained objects,  e.g. , graph and tree structures for part segmentation  [ 14 , 15 , 16 , 17 ] , and part learning in fine-grained recognition  [ 18 , 19 ] . In the era of deep learning, fine-grained segmentation and recognition approaches follow different paradigms, which achieve huge success compared to conventional models. Although there are more than 100 research papers each year to investigate this important problem, these papers seem to be disorganized, owing to various sundry research focuses including new task settings, benchmarking, and learning strategies. In particular, there are few survey papers that summarize the recent advances in fine-grained part segmentation. Thus the relationships among different fine-grained sub-tasks are still under-explored, and these sub-tasks are developed independently by regarding them as less-relevant tasks.

 
 
 In this paper, we make a comprehensive study of advances in fine-grained visual parsing tasks in the last decade. Besides analyzing recent deep learning works, we seek to explain the differences between non-deep learning and deep models, since these works often share similar intuitions and observations, and some of the previous studies could inspire further research. For consolidating these recent advances, we propose a new taxonomy for fine-grained part segmentation and recognition tasks, and also provide a collection of predominant benchmark datasets following our taxonomy. Besides these improvements compared to other survey papers  [ 20 , 21 ] , in this paper, we start from the novel view of  part relationship learning and regard it as the correspondences of different fine-grained sub-tasks. In this view, we revisit both the individual and universal challenges of part segmentation and fine-grained recognition and make an attempted solution using the guidance of  part relationship learning . In addition to these insights, we finish by discussing the future directions of fine-grained visual parsing tasks.

 
 
 To summarize, the main contributions of this survey are as follows:

 
 1. 
 
 We present a comprehensive survey of fine-grained visual parsing tasks by collecting recent advances of two representative tasks,  i.e. , semantic part segmentation and fine-grained recognition.

 

 2. 
 
 We revisit these fine-grained visual tasks from a novel perspective of part relationship learning, by revealing the connections of these fine-grained tasks and providing a promising solution to tackle challenges in fine-grained tasks.

 

 3. 
 
 We consolidate recent fine-grained research by reorganizing these works with new taxonomy, providing a collection of prevailing benchmark datasets, and make comprehensive discussions to inspire future works.

 

 4. 
 
 We provide promising future directions of fine-grained visual parsing tasks to inform further studies.

 

 
 
 
 Fig. 2: The landscape of semantic part segmentation tasks in our taxonomy. We summarize the recent advances from two different aspects: problem setting and learning strategy.
 
 
 
 The remainder of this paper is organized as follows: Section II provides new taxonomies, benchmark settings, and recent research on the problem of fine-grained part segmentation. Section III consolidates benchmarks, challenges, and advanced research on fine-grained object recognition. In Section IV , we delve into the connections of different fine-grained visual tasks from the perspective of part relationship learning and provide new solutions to improve existing challenges. We then highlight the future directions in Section V and then conclude this paper in Section VI .

 
 
 

## II Fine-grained Object Segmentation: A Part-level Perspective 

 

### II-A Taxonomy 

 
 In this section, we construct a new taxonomy for the semantic part segmentation task and revisit the fine-grained object segmentation from the part-level perspective. As in Fig. 2 , we conclude and re-organize the methods to solve semantic part segmentation tasks from two different views,  i.e. , the problem setting and learning strategies.

 
 
 1) Problem setting. Considering target objects to segment, we categorize these methods into two lines,  i.e. , single-class and multi-class parsing. The single-class part segmentation methods only tackle one specific category while objects of other categories should be taken as backgrounds. The multi-class setting aims to segment multiple classes that appear in the visual stream simultaneously. Regarding data collection in segmentation tasks, we further divide them into strongly-supervised, weakly-supervised and unsupervised learning. In similar ways, we consider the instance-level semantic-level, and video-based parsing problems. Given these terminologies defined here, some sub-areas show linguistic crosses with other ones,  e.g. , unsupervised learning with multi-class part parsing. However, as these sub-areas have not yet been explored, here we discuss the main branches that attract major research attention in Section II-B .

 
 
 2) Strategy. Learning to segment object parts has attracted a wide variety of research attention. In previous decades, several successful hand-crafted models have achieved success in segmenting objects with clear foreground representations,  i.e. , salient objects. We will briefly introduce several pioneer works in the following section. Besides these hand-crafted models, deep learning techniques have substantially improved the accuracy of segmentation models. We thus roughly group these techniques into three lines,  i.e. , pose-aided, multi-scale techniques, and using part relationships. Note that similar ideas could also be proposed in the non-deep learning methods. We will elaborate on their relations and differences in Section II-C .

 
 
 TABLE I: Summarization and comparisons of 12 widely-used part segmentation benchmark datasets. Note that PPS  [ 22 ] re-organizes two new datasets based on existing data annotations. 
 
 
 
 
 Dataset | 
 Pub. | 
 Year | 
 Task | 
 Image Num. | 
 Category | 
 Description | 

 
 
 
 Fashionista  [ 23 ] | 
 CVPR | 
 2012 | 
 Human Parsing | 
 685 | 
 56 | 
 Human
clothes parsing | 

 
 PASCAL-Part  [ 24 ] | 
 CVPR | 
 2014 | 
 Detection Segmentation | 
 10,103 | 
 NA | 
 First large-scale part segmentation dataset | 

 
 Horse-Cow dataset  [ 25 ] | 
 CVPR | 
 2015 | 
 Single-class Part Segmentation | 
 521 | 
 5 | 
 Quadruped animal parsing, reorganized from  [ 24 ] | 

 
 ATR  [ 26 ] | 
 ICCV | 
 2015 | 
 Human Parsing | 
 17,700 | 
 18 | 
 Human clothes parsing | 

 
 PASCAL-Person-Part  [ 27 ] | 
 CVPR | 
 2016 | 
 Human Parsing | 
 3,533 | 
 7 | 
 Human body parsing, reorganized from  [ 24 ] | 

 
 MHP  [ 28 ] | 
 arXiv | 
 2017 | 
 Human Parsing | 
 4,980 | 
 18 | 
 Multiple human clothes parsing | 

 
 LIP  [ 29 ] | 
 T-PAMI | 
 2018 | 
 Human Parsing | 
 50,462 | 
 20 | 
 Clothes parsing with human Poses | 

 
 VIP  [ 30 ] | 
 ACM MM | 
 2018 | 
 Video-based Human Parsing | 
 404 videos | 
 19 | 
 Video-based human clothes parsing | 

 
 CIHP  [ 31 ] | 
 ECCV | 
 2018 | 
 Instance-level Human Parsing | 
 38,280 | 
 20 | 
 Human clothes parsing | 

 
 PASCAL-Part-58  [ 32 ] | 
 CVPR | 
 2019 | 
 Multi-class Part Segmentation | 
 10,103 | 
 58 | 
 First multi-class dataset, reorganized from  [ 24 ] | 

 
 PPS  [ 22 ] | 
 CVPR | 
 2021 | 
 Part-aware Panoptic Segmentation | 
 10,103/3,475 | 
 194/23 | 
 Derived from VOC-2010/Cityscape dataset | 

 
 UDA-Part  [ 33 ] | 
 CVPR | 
 2022 | 
 Single-class Part Segmentation | 
 200 | 
 5 | 
 Unsupervised domain adaptation from synthetic vehicles | 

 
 ADE-20K-Part  [ 34 ] | 
 IJCV | 
 2022 | 
 Multi-class Part Segmentation | 
 10,103 | 
 544 | 
 Large scale multi-class dataset, reorganized from  [ 35 ] | 

 

 
 
 
 

### II-B Task Settings in Part Segmentation 

 
 Following the taxonomy in Section II-A , we first summarize these datasets according to the task settings and annotation labels. We then elaborate on the detailed task settings and popular methods to solve these problems.
According to the segmentation targets, here we summarize the popular datasets for semantic part segmentation tasks, which span the publications, image numbers, segmentation categories, and detailed descriptions.

 
 

#### II-B 1 Single-class Part Segmentation

 
 Human parsing. As in Tab. I , earlier works first tend to solve the specific categories of part segmentation,  i.e. , human parsing 1 1 
 1 
 
 
 
 Also noted as human part segmentation in some works. . Representative datasets including Fashionista  [ 23 ] focuses on the human clothes parsing, which segments human objects into typical classes including  shorts ,  shoes ,  boots and  sweaters . However, this dataset contains over 56 categories with a limited number of 685 images, which is not applicable to large machine-learning systems. With the development of deep learning techniques, large datasets are proposed to train and benchmark these deep models,  e.g. , ATR  [ 26 ] and LIP  [ 29 ] , which consist of over 50,000 images of 20 categories for training and testing. These large benchmarks, as well as the accompanying baseline, have achieved great success in parsing humans into dressing clothes. Nevertheless, decomposing human objects with different clothing parts would lead to semantic inconsistencies on certain occasions. Hence the other line of works proposes to segment human bodies into semantic parts following the morphological rules, which share the same definitions with human poses. For example, Chen  et al.   [ 27 ] propose to organize the PASCAL-Person-Part dataset to segment human bodies into 7 semantic parts, including lower/upper-arms, torsos, lower/upper-legs, heads, and backgrounds. Leading by this trend, dozens of works  [ 2 , 3 , 4 , 5 , 36 , 29 , 37 , 38 , 39 , 40 , 41 , 42 , 43 , 44 , 45 , 46 ] propose to address this critical issue using deep learning techniques, which build well-established parsing baselines for understanding human structures.

 
 
 The MHP dataset  [ 28 , 47 ] is presented to address multiple human parsing challenges that involve multiple human identities in one image, in addition to the conventional human parsing tasks. Beyond this challenging task, CIHP  [ 31 ] is established to solve the instance-level human parsing task,  e.g.   [ 3 , 48 , 49 , 50 , 51 , 52 , 53 , 54 ] , which not only requires the semantic information of foreground objects, but disentangles these parts into different human identities. Moreover, other works offer to investigate the part segmentation problem from a video-based perspective,  i.e. , video-based human parsing, with the proposal of VIP  [ 30 ] . Video-based part segmentation requires a semantic consistency of temporal sequences. In summary, the tasks of these new trends continue to be founded on the segmentation of human clothing, while ignoring human body structure exploration and leaving space for future research.

 
 
 Object part parsing. Except for human parsing tasks, here we review the object part segmentation task and divide the existing literature into two different lines: 1) rigid object part segmentation  [ 55 , 17 , 33 ] : including cars, aeroplanes, motorbikes, and other vehicles; 2) non-rigid object part segmentation  [ 25 , 56 , 57 ] : including birds, horses, cows, and other living creatures. Although these two lines of segmentation tasks can be uniformly solved by the popular deep learning schemes, challenges remain due to ambiguous semantic representations, blurriness around part boundaries, and anti-topological predictions. However, as the rigid objects usually consist of stable structures, hence Song  et al.   [ 55 ] and Liu  et al.   [ 33 ] proposed to embed the canonical 3D models into learnable 2D part segmentation tasks, while in  [ 17 ] , the static geometric relationships are calculated for deducting the related part regions. However, when extending these strategies to non-rigid objects, namely articulated objects, the connections between object parts show a significant variance because of the various part shapes. Hence the dynamic part relationships  [ 25 , 56 ] or pose-aided strategies  [ 57 ] are proposed to solve this problem, which will be elaborated on in the next subsection.

 
 
 Weakly-supervised part parsing. Several recent works have proposed exploiting weak supervision to generate dense semantic part segmentation masks. Unlike conventional weakly-supervised semantic segmentation methods, part segmentation using conventional image-level or box-level supervision would inevitably lead to highly repetitive semantic meanings (  e.g. , every human image contains the semantic label of torso and legs ) and ambiguous annotations (bounding box overlap) respectively. Thus several recent works propose to use human pose information as a weak supervision, which also shows high relationships with the part segmentation masks. We would like to elaborate on this in Section II-C2 . Wu  et al.   [ 58 ] and Yang  et al.   [ 59 ] propose to generate accurate part segmentation masks using keypoint annotations.
Moreover, Zhao  et al.   [ 60 ] propose a pose-to-part framework that gradually transfers weak pose annotations to the accurate segmentation masks and then use the image-level boundaries to correct the ambiguous regions.

 
 
 Unsupervised learning. The aforementioned part segmentation tasks require accurate pixel-level annotations. It is an extremely labor-consuming work  [ 61 ] , especially performing annotations in the fine-grained part levels. Hence another trend of works  [ 62 , 63 , 64 , 65 , 66 , 67 ] proposes to explore the semantic information through unsupervised manners. In  [ 62 ] , the automatic discovery of semantic parts and the relationships between the linguistical definition and activations discovered by CNNs are first explored. Leading by this thought, several works  [ 63 , 64 ] are proposed to leverage the advantages of deep representations. One specific feature is that the part representation after geometric transformation should be invariant over all instances of a category. Beyond this idea, Choudhury  et al.   [ 67 ] propose to discover the object part by using the contrastive loss among local regions. In addition, as all object instances from one category share the same part compositions, Gao  et al.   [ 65 ] propose to leverage the consistency of specific parts from different object instances, for example, different wings of birds share similar shapes and localization with respect to holistic objects.

 
 
 The above works undoubtedly demonstrate the strong ability of automatic part discovery by adding constraints to deep neural networks. Without any prior guidance, these localized parts show a strong relation to the morphological structures of object categories. Thus a natural concern arises: can we use this object compositional information to guide the learning of other tasks? Furthermore, with weak supervision ( e.g. , image-level class labels), can we localize the accurate part regions that are most helpful for the learning objective, and what are the relations among these parts in recognition tasks? Keeping these concerns in mind, we will explain and discuss these details in Section III .

 
 
 Fig. 3: Task settings of single-class and multi-class part segmentation. The single-class part segmentation only focuses on segmenting the objects of one specific class, while multi-class part segmentation aims to segment multiple objects that occurred in one scenario.
 
 
 
 

#### II-B 2 Multi-class Part Segmentation

 
 When revisiting the single-class part segmentation task in Fig. 3 b), there remain challenging problems for understanding the image content. Focusing on a single class,  e.g. , person class, and ignoring other meaningful classes,  e.g. , cars, and horses, would lead to severe parsing issues for understanding the context. In Fig. 3 , only parsing human bodies into parts leads to a lack of object interaction with the context, for example, what is the human doing and where is the human sitting?

 
 
 Multi-class semantic part segmentation. In response to these above challenges, the multi-class part segmentation tasks  [ 32 ] are naturally proposed, as in Fig. 3 c). The multi-class part segmentation tasks aim to segment objects of multiple classes into parts. In  [ 32 ] , a re-organized benchmark of the PASCAL-Part dataset is first proposed to solve this task, resulting in 58 semantic part classes. This new benchmark setting introduces additional challenges compared to the conventional single-class setting: 1) semantic ambiguity: parts of different object categories could share similar appearances,  e.g. , horse and cow legs; 2) boundary ambiguity: part boundaries of different objects are usually hard to disentangle. Toward this end, pioneer work  [ 32 ] proposes a joint boundary-semantic awareness framework with auxiliary supervision. In  [ 68 ] , a graph-based matching network is proposed to construct the complex relationships between different parts, handling the part-level ambiguity and localization problems, which achieves success in handling part segmentation tasks of large scales,  i.e. , 108 part classes.
While its journal version  [ 34 ] focuses on the improvements of edge localization and extends the ADE-20k dataset  [ 35 ] with part parsing labels, namely ADE 20K-Part.
Besides, Tan  et al.   [ 69 ] propose a semantic ranking loss to re-rank these semantic parts by their predicted confidence. Singh  et al.   [ 70 ] develop a new learning framework that increases scalability and reduces task complexity compared to the monolithic label space counterpart. Additionally, this new research  [ 70 ] introduces more complex part challenges,  i.e. , distinguishing left and right part localizations with more than 200 semantic classes.

 
 
 Panoptic part segmentation. Motivated by the Panoptic Segmentation proposed by  [ 71 ] , parsing objects into disjoint parts along with the background regions seems to construct a comprehensive interaction with the environmental context. Geus  et al.   [ 22 ] establish the Part-aware Panoptic Segmentation (PPS) task to understand a scene at multiple levels of abstraction. This PPS benchmark is founded on two representative datasets, PASCAL-VOC  [ 72 ] for daily images and Cityscapes  [ 73 ] for autonomous driving. In  [ 22 ] , a two-stage semantic parsing framework is proposed and the evaluation criteria for this new task are founded.

 
 
 
 

### II-C Strategies in Part segmentation 

 
 Beyond the specific challenges of task setting in Section II-B , part segmentation methods are designed following certain basic principles. Even the deep learning models and the non-deep ones share similar thoughts for constraining the optimization process. In this subsection, we will first introduce their commonalities and contrasts and then discuss and explore the promising future directions.

 
 

#### II-C 1 Non-deep learning models: hand-crafted priors

 
 Object part parsing in the past decade does not strictly follow the current definition of semantic segmentation, but decomposing a holistic object into basic compositional units shares the same concerns. In  [ 74 ] , the Deformable Part Models are proposed to localize and understand the whole object, which constructs a feature pyramid with respective deformative locations. Eslami  et al.   [ 75 ] propose a generative model to jointly learn the appearances and part shapes and use block-Gibbs Markov Chain Monte Carlo (MCMC) for fast inference.
Following this trend, Liu  et al.   [ 76 ] adopt the Markov random fields to model the color and appearance similarities, deciding the part belongings. Meng  et al.   [ 77 ] propose to initialize part seed proposals and then develop a seed propagation strategy to combine other potential regions. Some other researches  [ 78 , 79 , 6 ] segment object parts as an intermediate result to help the downstream tasks, including object detection, pose estimation, and action recognition.

 
 
 Besides these works, the other line of works proposed to build trees  [ 14 , 15 , 16 , 80 , 81 ] or graph models  [ 17 , 24 , 25 ] , depicting the relationships of different object parts . In  [ 14 ] , a joint bottom-up and top-down procedure is proposed to hierarchically decompose the holistic object into coarse parts, fine-grained parts, and basic lines/keypoints. Wang  et al.   [ 16 ] introduce hierarchical poselets, which decompose the human bodies into poselets ( e.g. , torso + left arm). Moreover, Several studies  [ 80 , 81 ] construct “And-Or” graphs to assemble the outputs of parts.  e.g. , Dong  et al.   [ 80 ] build a deformable mixture parsing model to simultaneously handle the deformation and multi-modalities of Parselets.
Other works resort to graph structures, which are relatively flexible compared to hierarchical trees. For example, Chen  et al.   [ 24 ] construct a relational graph by the part attributes itself and pair-wise relationships. Wang  et al.   [ 25 ] propose to learn the part compositional model under multiple viewpoints and poses, constructing a robust transformation of different conditions.

 
 
 Revisiting non-deep learning models. With the development of deep learning techniques  [ 82 , 83 , 84 ] , there is no doubt that the deep part segmentation models occupy the predominant places, benefited from their significant leading performance.
Following the end-to-end training framework  [ 85 ] in semantic segmentation, recent part segmentation models achieve more success than the conventional hand-crafted feature extractors,  e.g. , HOG or SIFT features. However, these deep learning models neglect the consideration of hierarchical body structures and would face great challenges in understanding unseen data and generating unreasonable segmentation masks. For example, in human parsing tasks, deep models always follow the statistical rules that deservedly take the round-shaped objects as human heads, which leads to incorrect parsing results for car wheels. In some error parsing cases, the lower legs could be connected with the upper bodies which breaks the basic topological rules. Interestingly, these phenomena are usually rare in conventional non-deep learning models, which follow the strict constraints of topological or morphological compositional principles,  e.g. , the human bodies are hierarchically decomposed into basic structures thus adjacent body parts show strong correlations. Moreover, the non-deep learning models require very little training data, showing great application potential in handling extreme circumstances in real-world applications.

 
 
 

#### II-C 2 Deep learning strategies

 
 In addition to these aforementioned differences, the non-deep learning and deep learning models share similar designs and basic foundations to solve the fine-grained part parsing tasks. Whether hand-crafted feature extractors or deep feature extractors are employed, the basic challenges still remain for parsing reasonable and clear segmentation results. Three important characteristics of deep learning-based models are discussed in this subsection.

 
 
 Pose-aided learning. The pose estimation and part segmentation are dual problems. Compared to the dense pixel prediction task of part segmentation, pose estimation is a more lightweight estimation task with significantly less annotation consumption. Conventional non-deep learning methods  [ 86 , 23 , 87 , 81 ] have proposed the importance of joint learning of these two related tasks. In the era of deep learning, major research efforts  [ 5 , 29 , 38 , 36 ] focus on the joint optimization of human part parsing and pose estimation with the proposals of large datasets. In  [ 38 ] , a mutual learning framework is proposed by embedding the dynamic kernel of pose estimation to part segmentations. Fang  et al.   [ 36 ] propose to transfer the human pose estimation knowledge as a coarse parsing prior and then to refine these coarse masks in the subsequent stages. Besides, other weakly-supervised methods  [ 60 ] using keypoints information also achieves notable successes.
Methods of this category verify that the accurate localizations of pose key points, including animals and human beings, shows strong benefits to the tasks of part segmentations.

 
 
 Multi-scale zooming. Different from the object segmentation tasks, part segmentation demonstrates a great demand in parsing detailed regions inside objects. Chen  et al.   [ 88 ] propose the atrous convolutional network to enhance the receptive field while  [ 89 ] introduces an improved structure of atrous spatial pyramid pooling (ASPP), which incorporates multi-scale features in one single layer. Besides these general improvements, in  [ 27 ] , a two-stream CNN is proposed to fuse the global features and local features. While Xia  et al.   [ 37 ] propose a stage-wise framework to detect and segment object parts from image-levels to object-levels and then part-levels.

 
 
 Part relationship guidance. Several recent works  [ 90 , 4 , 56 , 41 , 43 , 53 , 50 , 40 ] propose to embed the part-level relationship as learning priors to guide the segmentation process. For example, Wang  et al.   [ 56 ] propose a joint CRF to model the object-part and part-level relationships after the encoding of image features.
 [ 4 ] decouples the part segmentation learning as multiple independent tasks while using the part-level learning order to constrain the recurrent learning process. Gong  et al.   [ 41 ] adopt a universal graph learning strategy to model the part relationship across multiple datasets. Wang  et al.   [ 43 ] propose a hierarchical part parsing network to gradually decompose the object from the coarse level to the finer level.
In addition, in  [ 40 ] , a tree structure is constructed based on the CNN architectures and models the part-level relationship for understanding. Methods of this category successfully incorporate relationship learning to promote the segmentation process, while also using the accurate feature extraction of CNNs. The key challenge in fine-grained visual parsing is to understand the compositional relationships. Here we summarize these relationships as follows: 1) object-part relationship; 2) part-level relationship within one object; and 3) part-level across different images/objects.
By understanding these relationships, deep models can further promote the learning of action recognition, fine-grained object recognition, and re-identification tasks.

 
 
 
 
 

## III Fine-grained Object Recognition: Understanding Local Structures 

 

### III-A Definition and Challenges 

 
 Definition. Image Object classification has achieved great success benefiting from the development of deep learning systems and proposals of large datasets. Here we conclude the tasks of image object classification as base-level recognition , for example, classifying horses and aeroplanes, as in Fig. 1 . Objects in base-level categories can be easily distinguished by image-level global features and usually has large margins in semantic definitions. For fine-grained object recognition, deep learning systems are required to distinguish the subtle differences among sub-categories that have similar appearances and semantic definitions. In this problem, methods developed for base-level recognition usually face great challenges for classifying fine-grained classes as in Fig. 1 .

 
 
 The formulation of fine-grained object recognition is similar to the common base-level ones, by learning using a much more “compact” semantic label space.
Moreover, the generalized definition of fine-grained object recognition problems consists of two different levels of recognition: 1) subcategory-level: recognizing different fine-grained sub-categories that consist of multiple identities; 2) instance-level: distinguishing and identifying two different instances,  e.g. , person re-identification, vehicle re-identification, and face recognition. In this survey, we mainly focus on the first level of research but it should be noted that these two sub-field share many common techniques, which will be discussed in the following section.

 
 
 Fig. 4: Three typical challenges in fine-grained recognition tasks (images from CUB dataset  [ 91 ] ). 1) Heterogeneous semantic spaces: the semantic definitions of fine-grained text labels are usually cluster distributed. 2) Near-duplicated inter-class appearances: objects of different categories present visually similar appearances. 3) Inter-class shape variances: the shape structures of objects in the same category can be inconsistent.
 
 
 
 TABLE II: Summarization and comparisons of 13 widely-used part segmentation benchmark datasets. The Bbox and Part in annotation items indicate that the dataset provides object bounding box labels and part-level localization labels respectively. 
 
 
 
 
 Dataset | 
 Pub. | 
 Year | 
 Image Num. | 
 Category | 
 Annotation | 
 Description | 

 
 
 
 Oxford 102 Flowers  [ 92 ] | 
 ICCVGIP | 
 2008 | 
 8,189 | 
 102 | 
 - | 
 Flower Classification | 

 
 CUB-200-2011  [ 91 ] | 
 - | 
 2011 | 
 11,788 | 
 200 | 
 Bbox Part | 
 Birds, Best-known fine-grained benchmark | 

 
 Stanford Dogs  [ 93 ] | 
 CVPRW | 
 2011 | 
 20,580 | 
 120 | 
 Bbox | 
 Dog Classification | 

 
 Stanford Cars  [ 94 ] | 
 ICCVW | 
 2013 | 
 16,185 | 
 196 | 
 Bbox | 
 Car Classification | 

 
 FGVC Aircraft  [ 95 ] | 
 Arxiv | 
 2013 | 
 10,000 | 
 100 | 
 Bbox | 
 Aircraft Classification | 

 
 Food 101  [ 96 ] | 
 ECCV | 
 2014 | 
 101,000 | 
 101 | 
 - | 
 Food Classification | 

 
 BirdSnap  [ 97 ] | 
 CVPR | 
 2014 | 
 49,829 | 
 500 | 
 Bbox Part | 
 Large Bird datasets | 

 
 NAbirds  [ 98 ] | 
 CVPR | 
 2015 | 
 48,562 | 
 555 | 
 Bbox Part | 
 Large Bird datasets | 

 
 CompCars  [ 99 ] | 
 ECCV | 
 2018 | 
 136,727 | 
 431 | 
 Part Images | 
 Cars from web-nature and surveillance-nature | 

 
 DeepFashion  [ 100 ] | 
 CVPR | 
 2016 | 
 800,000 | 
 1,050 | 
 Bbox Part | 
 Clothes Classification | 

 
 iNat2017  [ 101 ] | 
 CVPR | 
 2018 | 
 857,877 | 
 5,089 | 
 Bbox | 
 Large-scale Species Classification | 

 
 Dogs-in-the-Wild  [ 102 ] | 
 ECCV | 
 2018 | 
 299,458 | 
 362 | 
 - | 
 Large-scale Dog Classitication | 

 
 iNat2021  [ 103 ] | 
 CVPR | 
 2021 | 
 3,286,843 | 
 10,000 | 
 - | 
 Improved version of iNat2017  [ 101 ] | 

 

 
 
 
 Challenges. Here we summarize three typical challenges of the fine-grained object recognition task in Fig. 4 : 1) Heterogeneous semantic space: although fine-grained labels are distributed in a compact space compared to the base-level category labels, their semantic definitions are still heterogeneous. For example, there are three types of blackbirds but only one bobolink in the semantic space. This phenomenon is still less-explored in the field of fine-grained recognition which leaves challenges for learning appropriate decision boundaries. 2) Near-duplicated inter-class appearances: in the middle of Fig. 4 , we present three images from different fine-grained categories while sharing much common ground in visual appearances. Thus, deep learning models need to clearly distinguish their differences by observing local details. 3) Intra-class shape variances: image objects that belong to the same categories can present in various shapes and structures. As for the bird classification task, the flying attitude shares less intuitive visual cues with that sitting one, bringing challenges to deducting these images with various shapes in the same categories.
In most cases, these three challenges show mutual effects on each other, and a good learning model needs to have the ability to handle semantic imbalance, inter-class similarities, and intra-class diversities simultaneously.

 
 
 

### III-B Benchmark Datasets 

 
 In this subsection, we summarize the prevailing benchmark datasets in the field of fine-grained recognition. In Tab. II , with the development of machine learning systems, earlier works have established benchmark datasets with more than 100 categories for the classification of common daily objects, including Oxford 102 Flowers  [ 92 ] for plants, CUB-200-2011  [ 91 ] for more than 200 bird categories, and Stanford-Dogs  [ 93 ] for the classification of 120 dog sub-categories.

 
 
 Pioneer machine learning methods, including SVM, and dictionary learning face great challenges in tackling these problems with less than 30% accuracies  [ 18 ] , indicating that these works cannot be directly used in real-world industrial applications. Thus to solve these problems, these datasets provide bounding box information for localizing the main objects and providing the box or segmentation masks for part learning,  e.g. , bird heads and torsos for the CUB dataset  [ 93 ] . Integrating this fore-ground information or part localization priors significantly helps the learning of fine-grained objects, especially the subtle differences of near-duplicated objects.

 
 
 With the development of deep learning systems, especially CNNs, the representation ability for fine-grained objects has been significantly improved, e.g. , from 28% to 75% accuracy on the CUB-200-2011 benchmark in  [ 6 ] . Despite their effectiveness, deep learning models usually rely on the acquisition of a large number of training data with similar distributions. Thus many new datasets with plentiful annotations are proposed, including Food 101  [ 96 ] with more than 101k images, NAbirds  [ 98 ] and BirdSnap  [ 97 ] for nearly 50k images.
These datasets not only provide high-quality annotations but also introduce new challenges for complicated semantic definitions, intra-class diversities, and inter-class similarities.

 
 
 Beyond these earlier deep learning benchmarks, recent advanced research proposes large-scale annotations with a huge number of fine-grained categories. For example, iNat2017  [ 101 ] provides more than 0.85M images of 5k categories, while its improved version iNat2021  [ 103 ] provides more than 3M images of 10k categories.
In addition, several other large-scale benchmarks  [ 104 , 105 ] have been proposed for fish recognition and landmark recognition.
Beyond the aforementioned challenges, these datasets span the new dilemmas: 1) imbalanced/long-tailed data distributions: objects of some rare categories usually consist of few annotations, while other main classes consist of thousands of images; 2) noisy labels: images of large scale datasets are usually collected webly and would introduce many noisy ambiguous labels. Thus the classification model needs to further purify these noisy factors by learning from predominant clean annotations.

 
 
 

### III-C Strategies in Fine-grained Recognition 

 
 Recognizing fine-grained objects has attracted much research attention in the last two decades. In this subsection, we first conclude the non-deep learning techniques including the hand-crafted features and human-in-loop learning frameworks in Section III-C1 . We then conclude the recent advanced research using deep learning techniques from two aspects,  i.e. , the part-guided learning in Section III-C2 and learning with feature representation constraints in Section III-C3 .

 
 

#### III-C 1 Non-deep learning Models

 
 Hand-crafted feature extraction. Pioneer fine-grained works  [ 106 , 107 , 18 , 108 ] propose to use hand-crafted features to recognize objects. For example, Zhang  et al.   [ 18 ] propose to incorporate the SVM into understanding pose structures, learning with SIFT and BoW (Bag of Words) features. Yao  et al.   [ 106 ] propose dense sampling strategies with random forests to extract local features. Other researches including  [ 107 ] adopt the codebook learning strategy for encoded dictionaries. Methods of this category can benefit from learning with local descriptors or part features while still facing difficulties in understanding fine-grained semantics.

 
 
 Human in the loop. Conventional machine learning methods usually lead to relatively low performance,  e.g. , 28% for CUB-200-2011 classification tasks, which have difficulties for applying in realistic applications. Hence earlier works propose to incorporate human expert knowledge into the learning process. For example, Wah  et al.   [ 109 ] leverage computer vision techniques and analyze the user responses to gradually enhance the final learning accuracies. While Branson  et al.   [ 19 ] propose an interactive scheme with deformable part models to distinguish the subtle differences between similar objects.

 
 
 

#### III-C 2 Part-guided Learning

 
 Supervised part learning. With the development of deep learning techniques, recognizing common base-level objects has made significant progress. Although the performance of fine-grained recognition has been improved in many applications  [ 6 ] , considering the challenges mentioned above, distinguishing subtle differences among near-duplicated objects usually faces serious dilemmas. Thus dozens of works propose to employ the part-level features  [ 6 , 7 , 8 , 9 , 110 , 111 , 10 , 112 ] to amplify these differences in a local perspective. Zhang  et al.   [ 6 ] propose a part-based R-CNN to locate the part features and then build pose-normalized features as the enhancement for global features. Krause  et al.   [ 113 ] propose to use keypoint annotations for fine-grained recognition, which leverages the co-segmentation techniques to align different views of images. Huang  et al.   [ 7 ] propose a dual-stream part-stacked CNN to jointly learn discriminative features from high-resolution part features and low-resolution global ones. In addition to these techniques using part detection methods, Wei  et al.   [ 9 ] propose to use part segmentation masks to regularize the local descriptor learning process. Although segmentation masks provide more accurate learning guidance, learning with all feasible part proposals would lead to a globally homogeneous amplification of every pixel.
To summarize, supervised part-learning techniques adopt the part detectors or segmentation masks as local feature selection guidance, and then fuse these local features with the image-level global ones. In this manner, not only the global features but the local details are taken into account for final feature distance measurements. However, these part excavation methods still rely on accurate part segmentation or detection annotations, requiring enormous labor consumption. In addition, considering that the accurate part annotations of test data are usually infeasible, transferring this expert part knowledge into the testing environment would also lead to an inductive bias, which would further limit the effectiveness in real-world applications.

 
 
 Fig. 5: Three typical high-order relations as in  [ 114 ] . Vanilla classification: encoded features are pooled into vectors for classification, used in most of the works. Second-order relationship  [ 115 , 116 , 117 , 118 , 114 ] : learning rich second-order features by keeping the spatial dimension. Trilinear attention  [ 119 , 120 , 121 ] : preserving the same size as input features for learning spatial-wise or channel-wise attention matrix.
 
 
 
 Unsupervised part learning. 2 2 
 2 
 
 
 
 also noted as weakly-supervised fine-grained recognition. 
Considering the labour-intensive computation and unstable generalization ability in the inference stage, recent ideas  [ 122 , 123 , 124 , 125 , 126 , 127 , 128 , 129 , 130 , 131 ] propose to use unsupervised part attention techniques. In  [ 62 ] , authors prove that during the back-propagation process, neural networks have the potential to discover semantic parts automatically. Leading by this trend, Simon  et al.   [ 122 ] propose neural activation constellations to localize semantic parts without any supervision. Different from this work, Fu  et al.   [ 124 ] propose a multi-stage zooming strategy to automatically locate and re-scale the attention regions, by learning the confidence scores of different zooming proposals. Similar to this work, Recasens  et al.   [ 125 ] develop a saliency-based sampling layer for neural networks after finding the activated regions. While Ge  et al.   [ 127 ] incorporate the weakly-supervised detection and segmentation models for localizing the discriminative features for fine-grained distinguishing. Besides, Wang  et al.   [ 126 ] propose a Gaussian mixture model for investigating the object parts with an auxiliary branch for supervision. In  [ 131 ] , a graph-propagation correlation learning method is proposed to model and propagate the discriminative part features to other parts. Nevertheless, these methods have shortcomings in two aspects: 1) introducing auxiliary learning branches or stages for optimization; and 2) the number of part proposals can sometimes be large whereas only a few are useful for recognition.

 
 
 To solve this issue as well as reduce computational costs, Lam  et al.   [ 132 ] propose an HSNet searching architecture to explore the most discriminative parts, while other work  [ 8 ] builds a weakly-supervised part selection mechanism based on their response scores. Zhao  et al.   [ 133 ] propose a Transformer architecture to build inter-part relationships and adopt multiple auxiliary branches for part-awareness learning, while in the inference stage, these auxiliary branches are not used for computational consideration.
Methods of unsupervised part learning  [ 134 , 135 , 133 , 136 ] lead the prevailing trend in fine-grained recognition, which benefits from its strong ability in understanding local differences and discovering object parts. Furthermore, selecting and modeling the part relationships becomes an emerging topic in fine-grained recognition.

 
 
 Different from the unsupervised part learning in semantic segmentation, part attention in fine-grained recognition aims to discover the discriminative features and exploits these local features as an enhancement for distinguishing near-duplicated objects. Thus the semantic information of the unsupervised part in recognition is usually not strictly aligned with the natural common definitions.

 
 
 

#### III-C 3 Feature Representation Learning

 
 Besides the methods using part-level features for enhancing the local details, the other crucial problem in fine-grained recognition is feature representation learning. There is intuitive thinking that well-represented features can provide a more robust and generalization ability for downstream tasks, including segmentation, detection, and also fine-grained recognition. Despite the experimental evidence, enhancing the detailed representation ability helps the measurement of local subtle differences hidden among different features, which may be vital factors for discrimination. When features of various images are distributed in one generalized and robust fine-grained feature space, these subtle differences would be easy to discover. Guided by the theory, this line of methods tends to regularize the feature learning process  [ 137 , 138 , 139 , 140 , 141 , 142 ] or generate rich feature representations  [ 115 , 116 , 117 , 118 , 143 , 144 , 119 , 120 , 145 , 114 ] without using additional annotations.

 
 
 High-order representations. As in Fig. 5 a), given an input image ℐ \mathcal{I} , the conventional classification model can be represented as 𝐗 = Φ ⁡ ( ℐ ) \mathbf{X}=\Phi(\mathcal{I}) . Thus 𝐗 ∈ ℝ W × H × C \mathbf{X}\in\mathbb{R}^{W\times H\times C} denotes the C C -dimensional with H × W H\times W feature maps and the final classification vector would be Pool ​ ( 𝐗 ) ∈ ℝ 1 × 1 × C \bm{\texttt{Pool}}(\mathbf{X})\in\mathbb{R}^{1\times 1\times C} . Considering the spatial feature relationship is neglected during the pooling operations, high-order interactions are proposed in advanced works. In Fig. 5 b), as the pioneer work using the second-order relationship, Lin  et al.   [ 115 ] propose a bilinear model to extract shape and appearances by two different CNNs and then construct a bilinear pooling operation to generate rich second-order representations,  i.e. , 𝐗 = 1 W ​ H ​ ∑ i = 1 W ∑ j = 1 H vec ​ ( Φ 1 ​ ( ℐ ) i , j ⊤ ​ Φ 2 ​ ( ℐ ) i , j ) \mathbf{X}=\frac{1}{WH}\sum_{i=1}^{W}\sum_{j=1}^{H}\bm{\texttt{vec}}(\Phi_{1}(\mathcal{I})^{\top}_{i,j}\Phi_{2}(\mathcal{I})_{i,j}) . Although this bilinear pooling operation enriches the fine-grained representation and amplifies the differences of similar embedding, it also introduces high computational costs.  i.e. , C × C × N c ​ l ​ s C\times C\times N_{cls} for optimization, and N c ​ l ​ s N_{cls} denotes the category numbers. To solve this,
Gao  et al.   [ 116 ] propose a compact bilinear pooling model that uses the network itself to build second-order relationships,  i.e. , 𝐗 ≡ 𝐘 \mathbf{X}\equiv\mathbf{Y} . Other works propose to use matrix factorization  [ 117 ] , Grassmann constraints  [ 118 ] , and low-rank learning  [ 146 ] to reduce computation costs. Besides these works, Yu  et al.   [ 144 ] propose a hierarchical feature interaction operation to build heterogeneous second-order relationships. Zhao  et al.   [ 114 ] propose a graph-based high-order relationship learning to reduce the high-dimension space into discriminative low dimensions.

 
 
 However, the second-order feature learning still introduces the curse of dimensionality for optimization, thus the other line of works proposes to use the third-order relationship in Fig. 5 c), namely trilinear attention  [ 119 , 120 ] or non-local mechanisms  [ 121 ] . For example, Zheng  et al.   [ 119 ] propose the trilinear attention in the channel-dimension with a distillation mechanism, which can be formulated as: softmax ​ ( 𝐗 ⊤ ​ 𝐗 ) ​ 𝐗 ⊤ ∈ ℝ W ​ H × C \bm{\texttt{softmax}}(\mathbf{X}^{\top}\mathbf{X})\mathbf{X}^{\top}\in\mathbb{R}^{WH\times C} . While Gao  et al.   [ 120 ] propose a contrastive loss to learn the channel-wise relationship of inter-and intra-images. Methods using the third-order relationship maintain the output size and can be embedded into different network stages to enhance the representations.

 
 
 Fig. 6: Part relationship learning in two representative fine-grained visual tasks, fine-grained part segmentation  [ 32 ] and fine-grained recognition  [ 114 ] . First row: understanding complex fine-grained images requires the accurate parsing of local part relationships.
Second row: the cross-object relationship uses contextual information to help the understanding of small parts, while the object internal relationship with other parts helps the distinguishing of locally similar regions. Besides, the segmentation results can serve as parsing guidance and relationship learning in both tasks constructing the robust local structure understanding.
 
 
 
 Feature interactions and regularization. Besides building high-order rich features, the other line of works proposes to use feature interactions  [ 137 , 102 , 147 , 148 , 149 , 150 , 151 , 152 ] or using additional constraints  [ 153 , 154 , 138 , 139 , 140 , 155 ] . Wang  et al.   [ 137 ] construct a discriminative feature bank of convolutional filters that captures class-specific discriminative patches. Sun  et al.   [ 102 ] propose a multi-attention multi-constraint network to regularize the feature distributions based on the selected anchors. While Luo  et al.   [ 148 ] propose to learn cross-level and cross-images relationships for building interactive feature representations. For robust feature learning, Chen  et al.   [ 147 ] incorporate an additional destruction and construction branch as an additional learning task.
These works rely on additional blocks or feature interaction networks, which may introduce additional computation costs.

 
 
 Besides works using additional parameters to feature enhancement, several works propose to use auxiliary constraints in addition to the basic cross-entropy constraints. In  [ 138 ] , pair-wise confusion is proposed among Siamese networks to alleviate the overfitting issues. While Dubey  et al.   [ 140 ] propose an entropy maximizing approach to regularize the final classification confidence. Aodha  et al.   [ 139 ] propose geographically guided loss functions that deduct the fine-grained features using temporal and geographical spatial priors.
Besides, other works  [ 153 , 154 , 156 ] follow a self-supervised learning trend for fine-grained recognition. Wu  et al.   [ 153 ] propose to solve the dilemma between self-supervised learning and fine-grained recognition by enhancing the salient foreground regions.

 
 
 To summarize, 1) methods using high-order relations modules mainly focus on rich representations at the feature-level, and enhancing these representations would amplify the subtle differences among different object features, and 2) methods using additional constraints make fine-grained features to be distributed in compact and precise spaces, while alleviating the overfitting issue and concentrating more on object regions. This overfitting issue is further studied in existing works by generating accurate class activation maps while preventing only focusing on the local part regions. In the next section, we will discuss why we need local details and why only local details cannot perform accurate fine-grained recognition.

 
 
 
 
 

## IV Part Relationship in Segmentation and Recognition 

 
 Fine-grained visual parsing, including recognition, segmentation, detection and other high-level image understanding tasks, leaves us with challenges in its detailed and complex “fine-grained” parsing requirements. Understanding images with fine-grained objects can be substantially different from common “coarse-grained” ones. In this section, we investigate two representative fine-grained visual tasks,  i.e. , segmentation and recognition with the following natural concerns:

 
 1. 
 
 What are the key challenges in fine-grained recognition, or what are the unique problems in this subfield?

 

 2. 
 
 Why does part relationship learning help the understanding of these fine-grained tasks? What are the relations among them?

 

 
 
 

### IV-A Problems in Fine-grained Parsing 

 
 Fine-grained visual parsing is a relatively-defined concept compared to the common daily categories. Considering the specific tasks of fine-grained recognition and semantic part segmentation, understanding image objects would face the following challenges.

 
 
 1) Non-salient/less-prominent in image-level: the fine-grained visual features are imperceptible using existing learning systems or not easily understood by human visual systems. In fine-grained recognition, objects of different semantic classes usually share visually similar appearances but still show imperceptible discrepancies. This means that objects of these categories are recognizable by detailed local differences while understanding them using coarse global features is impracticable.
In the task of fine-grained part segmentation, distinguishing different parts relies on subtle visual cues, including imperceptible part boundaries and near-duplicated local visual patterns. The term imperceptible here means that these part boundaries can be relatively non-significant compared to object silhouettes and even contextual noisy information. In addition, in some extreme cases, due to the small scale of object parts (in Fig. 6 a)), it is difficult to recognize them by only observing small objects themselves. And in some cases, these discriminative cues in both parsing tasks are relatively non-significant, and are suppressed in the image-level feature representations.

 
 
 2) Locally distinguishable and indistinguishable: as mentioned in the first challenge, the fine-grained objects are only recognizable in local details,  e.g. , Fig. 4 in Section III . Thus enhancing the representation of these local parts is beneficial to learn discriminative embedding. However, these local part regions are not always distinguishable, for example, birds of two different categories may exhibit similar appearances in torso, tail, and wings while only differing in their heads. Analogous to the fine-grained recognition task, we present a similar segmentation problem in Fig. 6 b). The legs of the horse and cow are locally indistinguishable whereas the holistic object categories are easy to recognize and segment, since their head regions are salient for distinguishing. To sum up, the local regions of an image may be distinguishable or indistinguishable when compared with different images.

 
 
 3) Ambiguous semantic definition: the last challenge in fine-grained parsing is the ambiguous semantic definition, which is less explored in recent works. Conventional part parsing works  [ 14 , 15 ] propose to build hierarchical structures,  e.g. , from the holistic body to object parts, and then to line segments. However, considering the goal of segmentation and recognition, the semantic definitions of “fine-grained” tasks become an increasingly critical problem. For segmentation tasks, LIP dataset  [ 29 ] defines the human bodies with different fashion clothes,  e.g. , skirts and coats, while other datasets  [ 24 ] tend to segment bodies with the morphological rules,  i.e. , upper and lower bodies. Similarly, if we define two different categories of objects,   e.g. , bulldogs and poodles , the bulldogs can be subdivided into English bulldogs and American bulldogs with fewer differences. Thus the definitions of fine-grained semantics leave us with severe challenges, or in other words, “how fine is fine-grained parsing”?

 
 
 Fig. 7: Correlations schematic of three relevant tasks,  i.e. , part relationship representation, part segmentation and fine-grained recognition.
 
 
 
 

### IV-B Part Relationship Learning: A Solution 

 
 Towards the aforementioned challenges, in this survey, we argue that building Part Relationships in fine-grained visual parsing would be one reliable and promising solution. Here we elaborate on its effectiveness in solving these challenges: 1) considering that the fine-grained cues are usually hidden in local details and cannot be distinguished using the image-level features. Hence enhancing the learning process with part relations helps a dynamic understanding, as illustrated in the first line of Fig. 6 by  [ 114 ] . It should be mentioned that most of the prevailing works in fine-grained parsing and even re-identification tend to amplify fixed part regions of the same image, which could be solved by introducing part relationships. 2) Given one image, we usually do not understand which part should network focus on and what are discriminative features for recognition and segmentation. Thus understanding part relationships helps this problem in many ways,  e.g. , cross-part relationships within each object helps the understanding of geometric structures, object-part relationship helps the distinguishing of locally similar visual patterns, and part relationships across different images and objects help to learn the semantic consensus. Besides, considering the comparison of visually similar images, these different part-level relationships help the dynamic enhancement of feature extractions. 3) For the semantic ambiguities of fine-grained definition, here we advocate the learning with hierarchical structures, which are still less explored in deep learning works. For example, the human body can be decomposed into heads and bodies, while the head regions can be further subdivided into faces, eyes, and other organs. Building hierarchical trees or graphs helps the holistic geometric structures be more reasonable and is also beneficial for handling the heterogeneous semantic definitions as mentioned in Fig. 4 .

 
 
 Relations to fine-grained tasks. What roles does the part relationship learning play in different fine-grained understanding tasks? Here we present a schematic in Fig. 7 with fine-grained part segmentation, fine-grained recognition and part relationship representations. Part segmentation is the subset of the part relationship representations, while the latter consists of other structural parsing and detection sub-tasks. The intersections of part segmentation and fine-grained recognition are also illustrated in Section III-C2 . Methods of this category tend to utilize the part priors to guide the feature extraction, including the fine-grained classification  [ 6 , 112 , 7 , 8 , 9 , 10 ] and re-identifications  [ 110 , 157 , 158 , 159 , 160 ] . Understanding part relationships without segmenting them as explicit masks or boxes also considerably facilitates the distinguishing of subtle differences, considering the joint region of part relationship representations and fine-grained recognition. Methods using this idea usually adopt the attention mechanisms  [ 122 , 62 , 125 , 124 , 127 , 126 , 132 , 8 , 133 ] or graph-based structures  [ 134 , 135 ] to guide the learning process. Besides, the other line of works  [ 137 , 138 , 139 , 140 , 115 , 116 , 117 , 118 , 144 , 119 , 120 , 114 ] does not rely on part relationships and focuses on the feature representation learning process as mentioned above. It should be noted here we extend the concept of part relationship learning, by incorporating learning with implicit and explicit part localizations, and those methods build explicit part-level relationships. Moreover, fine-grained part segmentation is one of the explicit ways to understand the part relationship procedure, and learning in hierarchical manners or tree/graph structures also leads in promising directions.

 
 
 
 

## V Future Directions 

 
 Despite the significant progress made by existing works, there are still many unsolved problems in fine-grained recognition. Here we propose several promising future directions for discussion.

 
 
 Dynamic part relationship learning. As discussed in Section IV-B , prevailing part relationship learning works focus on building static connections and responses. For example, in fine-grained recognition tasks, networks tend to amplify fixed regions of the same image. Although this helps the discriminative embedding in training data distributions, it faces great challenges when compared to unknown novel testing examples. Besides, the dynamic relationships help the understanding of semantic parsing when objects face occlusions or abnormal gestures.

 
 
 Few-shot fine-grained learning. The learning of fine-grained classification is based on sufficient training data, while few-shot learning  [ 161 , 162 , 163 ] is nowadays an attractive trend for understanding novel concepts with only a few annotated images. As for the few-shot fine-grained classification, there are several advances  [ 164 , 165 , 166 , 167 , 168 ] in this field. However, these works usually follow the N-way K-shot trend, and N is usually set as 5 for the number of categories, indicating the huge gap compared to popular datasets with 500 to 5,000 categories. Besides, few-shot learning also can be regarded as a long-tailed setting where several categories are labeled with sufficient images but a few categories with limited annotations. This few-shot/long-tailed setting is a more realistic challenge that can be taken as a promising direction.

 
 
 Hierarchical structures of fine-grained learning. In Fig. 4 and Section IV-A , it is noted that the definition of fine-grained settings still exists ambiguous and some subcategories can be further divided into finer levels. Thus to solve this imbalance in semantic space and visual space, we argue for developing hierarchical structures to gradually subdivide these components into meaningful leaf nodes. These leaf nodes can be presented using basic units including pixels and line segments but also basic structures,  i.e. , organs of human beings. In other words, the hierarchical structures help to maintain similar concepts in semantic space aligned with that of visual spaces. Several pioneer works  [ 169 , 170 , 155 ] have explored the tree structures or hyper classes for fine-grained semantic structures. However, how to unify the semantic language embedding with the image-level visual features is still an under-explored problem. One promising direction is to unify the language and visual spaces using contrastive learning and mask modeling, including CLIP  [ 171 ] , GLIP  [ 172 ] , and other multi-modality learning methods.

 
 
 3D-aware fine-grained learning. In addition to the aforementioned 2D-based learning mechanisms, the other promising direction is 3D-aware fine-grained learning. In the semantic parsing of 3D models, many research efforts  [ 173 , 174 , 175 , 176 ] have been proposed to parse 3D objects with point cloud, mesh and voxel representations. Thus an interesting question arises here: what is the relationship between 3D parsing models the real-world 2D images? Earlier works collected in  [ 177 ] propose to use 3D models to aid the recognition of human faces by hand-crafted filters or template learning techniques. In the era of deep learning, several works  [ 178 ] propose to embed the 3D canonical model with learnable warping parameters to represent diverse 2D images. The emerging field can be further boosted with learnable mechanisms including the Neural rendering field  [ 179 ] .

 
 
 

## VI Conclusions 

 
 In this paper, we present a comprehensive survey of fine-grained visual parsing tasks from the novel perspective of part relationship learning. In this view, we delve into the connections of two representative fine-grained tasks,  i.e. , fine-grained recognition and part segmentation, and propose a new taxonomy to reorganize recent research advances including the conventional methods and deep learning methods. By consolidating these works and popular benchmarks, we propose the universal challenges left in fine-grained visual parsing and make an attempted solution from the view of part relationship learning. Besides, we also point out several promising research directions that can be further explored. We hope these contributions will provide new inspiration to inform future research in the field of fine-grained visual parsing.

 
 
 

## Acknowledgment

 
 This work was supported in part by the Key-Area Research and Development Program of Guangdong Province under Contract 2021B0101400002, the National Natural Science Foundation of China under contracts No. 62132002, No. 61825101, No. 62202010 and and also supported by the China Postdoctoral Science Foundation No. 2022M710212.

 
 
 

## References

 
 
 [1] 
 
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A
large-scale hierarchical image database,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , 2009, pp. 248–255.

 

 
 [2] 
 
X. Liang, X. Shen, D. Xiang, J. Feng, L. Lin, and S. Yan, “Semantic object
parsing with local-global long short-term memory,” in IEEE Conference
on Computer Vision and Pattern Recognition (CVPR) , 2016, pp. 3185–3193.

 

 
 [3] 
 
T. Ruan, T. Liu, Z. Huang, Y. Wei, S. Wei, and Y. Zhao, “Devil in the details:
Towards accurate single and multiple human parsing,” in AAAI
Conference on Artificial Intelligence (AAAI) , 2019, pp. 4814–4821.

 

 
 [4] 
 
Y. Zhao, J. Li, Y. Zhang, Y. Song, and Y. Tian, “Ordinal multi-task part
segmentation with recurrent prior generation,” IEEE transactions on
pattern analysis and machine intelligence , vol. 43, no. 5, pp. 1636–1648,
2021.

 

 
 [5] 
 
F. Xia, P. Wang, X. Chen, and A. L. Yuille, “Joint multi-person pose
estimation and semantic part segmentation,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , 2017, pp. 6769–6778.

 

 
 [6] 
 
N. Zhang, J. Donahue, R. Girshick, and T. Darrell, “Part-based r-cnns for
fine-grained category detection,” in European Conference on Computer
Vision (ECCV) . Springer, 2014, pp.
834–849.

 

 
 [7] 
 
S. Huang, Z. Xu, D. Tao, and Y. Zhang, “Part-stacked cnn for fine-grained
visual categorization,” in IEEE Conference on Computer Vision and
Pattern Recognition (CVPR) , 2016, pp. 1173–1182.

 

 
 [8] 
 
X. He and Y. Peng, “Weakly supervised learning of part selection model with
spatial constraints for fine-grained image classification,” in AAAI
Conference on Artificial Intelligence (AAAI) , 2017.

 

 
 [9] 
 
X.-S. Wei, C.-W. Xie, J. Wu, and C. Shen, “Mask-cnn: Localizing parts and
selecting descriptors for fine-grained bird species categorization,”
 Pattern Recognition , vol. 76, pp. 704–714, 2018.

 

 
 [10] 
 
Z. Huang and Y. Li, “Interpretable and accurate fine-grained recognition via
region grouping,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2020, pp. 8662–8672.

 

 
 [11] 
 
V. T. Bickel, J. Aaron, A. Manconi, S. Loew, and U. Mall, “Impacts drive lunar
rockfalls over billions of years,” Nature communications , vol. 11,
no. 1, pp. 1–7, 2020.

 

 
 [12] 
 
X. Sun, P. Wang, Z. Yan, F. Xu, R. Wang, W. Diao, J. Chen, J. Li, Y. Feng,
T. Xu et al. , “Fair1m: A benchmark dataset for fine-grained object
recognition in high-resolution remote sensing imagery,” ISPRS Journal
of Photogrammetry and Remote Sensing , vol. 184, pp. 116–130, 2022.

 

 
 [13] 
 
D. Pakhomov, V. Premachandran, M. Allan, M. Azizian, and N. Navab, “Deep
residual learning for instrument segmentation in robotic surgery,” in
 International Workshop on Machine Learning in Medical Imaging . Springer, 2019, pp. 566–573.

 

 
 [14] 
 
L. L. Zhu, C. Lin, H. Huang, Y. Chen, and A. Yuille, “Unsupervised structure
learning: Hierarchical recursive composition, suspicious coincidence and
competitive exclusion,” in European Conference on Computer Vision
(ECCV) , 2008, pp. 759–773.

 

 
 [15] 
 
J.-W. Hsieh, C.-H. Chuang, S.-Y. Chen, C.-C. Chen, and K.-C. Fan,
“Segmentation of human body parts using deformable triangulation,”
 IEEE Transactions on Systems, Man, and Cybernetics-Part A: Systems and
Humans , vol. 40, no. 3, pp. 596–610, 2010.

 

 
 [16] 
 
Y. Wang, D. Tran, and Z. Liao, “Learning hierarchical poselets for human
parsing,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) . IEEE, 2011, pp.
1705–1712.

 

 
 [17] 
 
A. Y. Wenhao Lu, Xiaochen Lian, “Parsing semantic parts of cars using
graphical models and segment appearance consistency,” in British
Machine Vision Conference (BMVC) , 2014.

 

 
 [18] 
 
N. Zhang, R. Farrell, and T. Darrell, “Pose pooling kernels for sub-category
recognition,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) . IEEE, 2012, pp.
3665–3672.

 

 
 [19] 
 
S. Branson, P. Perona, and S. Belongie, “Strong supervision from weak
annotation: Interactive training of deformable part models,” in IEEE
International Conference on Computer Vision (ICCV) . IEEE, 2011, pp. 1832–1839.

 

 
 [20] 
 
B. Zhao, J. Feng, X. Wu, and S. Yan, “A survey on deep learning-based
fine-grained object classification and semantic segmentation,”
 International Journal of Automation and Computing , vol. 14, no. 2, pp.
119–135, 2017.

 

 
 [21] 
 
X.-S. Wei, Y.-Z. Song, O. Mac Aodha, J. Wu, Y. Peng, J. Tang, J. Yang, and
S. Belongie, “Fine-grained image analysis with deep learning: A survey,”
 IEEE Transactions on Pattern Analysis and Machine Intelligence , 2021.

 

 
 [22] 
 
D. de Geus, P. Meletis, C. Lu, X. Wen, and G. Dubbelman, “Part-aware panoptic
segmentation,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2021, pp. 5485–5494.

 

 
 [23] 
 
K. Yamaguchi, M. H. Kiapour, L. E. Ortiz, and T. L. Berg, “Parsing clothing in
fashion photographs,” in IEEE Conference on Computer Vision and
Pattern Recognition (CVPR) , 2012, pp. 3570–3577.

 

 
 [24] 
 
X. Chen, R. Mottaghi, X. Liu, S. Fidler, R. Urtasun, and A. Yuille, “Detect
what you can: Detecting and representing objects using holistic models and
body parts,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2014, pp. 1971–1978.

 

 
 [25] 
 
J. Wang and A. L. Yuille, “Semantic part segmentation using compositional
model combining shape and appearance,” in IEEE Conference on Computer
Vision and Pattern Recognition (CVPR) , 2015, pp. 1788–1797.

 

 
 [26] 
 
X. Liang, C. Xu, X. Shen, J. Yang, S. Liu, J. Tang, L. Lin, and S. Yan, “Human
parsing with contextualized convolutional neural network,” in IEEE
International Conference on Computer Vision (ICCV) , 2015, pp. 1386–1394.

 

 
 [27] 
 
L.-C. Chen, Y. Yang, J. Wang, W. Xu, and A. L. Yuille, “Attention to scale:
Scale-aware semantic image segmentation,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , 2016, pp. 3640–3649.

 

 
 [28] 
 
J. Li, J. Zhao, Y. Wei, C. Lang, Y. Li, T. Sim, S. Yan, and J. Feng,
“Multiple-human parsing in the wild,” arXiv preprint
arXiv:1705.07206 , 2017.

 

 
 [29] 
 
X. Liang, K. Gong, X. Shen, and L. Lin, “Look into person: Joint body parsing
 pose estimation network and a new benchmark,” IEEE Transactions on
Pattern Analysis and Machine Intelligence , 2018.

 

 
 [30] 
 
Q. Zhou, X. Liang, K. Gong, and L. Lin, “Adaptive temporal encoding network
for video instance-level human parsing,” in ACM International
Conference on Multimedia , 2018, pp. 1527–1535.

 

 
 [31] 
 
K. Gong, X. Liang, Y. Li, Y. Chen, M. Yang, and L. Lin, “Instance-level human
parsing via part grouping network,” in European Conference on Computer
Vision (ECCV) , 2018, pp. 770–785.

 

 
 [32] 
 
Y. Zhao, J. Li, Y. Zhang, and Y. Tian, “Multi-class part parsing with joint
boundary-semantic awareness,” in IEEE International Conference on
Computer Vision (ICCV) , 2019.

 

 
 [33] 
 
Q. Liu, A. Kortylewski, Z. Zhang, Z. Li, M. Guo, Q. Liu, X. Yuan, J. Mu,
W. Qiu, and A. Yuille, “Learning part segmentation through unsupervised
domain adaptation from synthetic vehicles,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , 2022.

 

 
 [34] 
 
U. Michieli and P. Zanuttigh, “Edge-aware graph matching network for
part-based semantic segmentation,” International Journal of Computer
Vision , vol. 130, no. 11, pp. 2797–2821, 2022.

 

 
 [35] 
 
B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, and A. Torralba, “Scene
parsing through ade20k dataset,” in IEEE Conference on Computer Vision
and Pattern Recognition (CVPR) , 2017, pp. 633–641.

 

 
 [36] 
 
H.-S. Fang, G. Lu, X. Fang, J. Xie, Y.-W. Tai, and C. Lu, “Weakly and semi
supervised human body part parsing via pose-guided knowledge transfer,” in
 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) ,
2018, pp. 70–78.

 

 
 [37] 
 
F. Xia, P. Wang, L.-C. Chen, and A. L. Yuille, “Zoom better to see clearer:
Human and object parsing with hierarchical auto-zoom net,” in European
Conference on Computer Vision (ECCV) , 2016, pp. 648–663.

 

 
 [38] 
 
X. Nie, J. Feng, and S. Yan, “Mutual learning to adapt for joint human parsing
and pose estimation,” in European Conference on Computer Vision
(ECCV) , 2018, pp. 502–517.

 

 
 [39] 
 
J. Li, J. Zhao, C. Lang, Y. Li, Y. Wei, G. Guo, T. Sim, S. Yan, and J. Feng,
“Multi-human parsing with a graph-based generative adversarial model,”
 ACM Transactions on Multimedia Computing, Communications, and
Applications (TOMM) , vol. 17, no. 1, pp. 1–21, 2021.

 

 
 [40] 
 
W. Wang, Z. Zhang, S. Qi, J. Shen, Y. Pang, and L. Shao, “Learning
compositional neural information fusion for human parsing,” in IEEE
International Conference on Computer Vision (ICCV) , October 2019.

 

 
 [41] 
 
K. Gong, Y. Gao, X. Liang, X. Shen, M. Wang, and L. Lin, “Graphonomy:
Universal human parsing via graph transfer learning,” in IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) , 2019, pp.
7450–7459.

 

 
 [42] 
 
X. Liu, M. Zhang, W. Liu, J. Song, and T. Mei, “Braidnet: Braiding semantics
and details for accurate human parsing,” in ACM International
Conference on Multimedia , 2019, pp. 338–346.

 

 
 [43] 
 
W. Wang, H. Zhu, J. Dai, Y. Pang, J. Shen, and L. Shao, “Hierarchical human
parsing with typed part-relation reasoning,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , June 2020.

 

 
 [44] 
 
T. Zhou, W. Wang, S. Liu, Y. Yang, and L. Van Gool, “Differentiable
multi-granularity human representation learning for instance-aware human
semantic parsing,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2021, pp. 1622–1631.

 

 
 [45] 
 
D. Zeng, Y. Huang, Q. Bao, J. Zhang, C. Su, and W. Liu, “Neural architecture
search for joint human parsing and pose estimation,” in IEEE
International Conference on Computer Vision (ICCV) , 2021, pp.
11 385–11 394.

 

 
 [46] 
 
Y. Liu, S. Zhang, J. Yang, and P. Yuen, “Hierarchical information passing
based noise-tolerant hybrid learning for semi-supervised human parsing,” in
 AAAI Conference on Artificial Intelligence (AAAI) , vol. 35, no. 3,
2021, pp. 2207–2215.

 

 
 [47] 
 
J. Zhao, J. Li, H. Liu, S. Yan, and J. Feng, “Fine-grained multi-human
parsing,” International Journal of Computer Vision , vol. 128, no. 8,
pp. 2185–2203, 2020.

 

 
 [48] 
 
L. Yang, Q. Song, Z. Wang, and M. Jiang, “Parsing r-cnn for instance-level
human analysis,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , June 2019.

 

 
 [49] 
 
P. Li, Y. Xu, Y. Wei, and Y. Yang, “Self-correction for human parsing,”
 IEEE Transactions on Pattern Analysis and Machine Intelligence , 2020.

 

 
 [50] 
 
H. He, J. Zhang, Q. Zhang, and D. Tao, “Grapy-ml: Graph pyramid mutual
learning for cross-dataset human parsing,” in AAAI Conference on
Artificial Intelligence (AAAI) , vol. 34, no. 07, 2020, pp. 10 949–10 956.

 

 
 [51] 
 
R. Ji, D. Du, L. Zhang, L. Wen, Y. Wu, C. Zhao, F. Huang, and S. Lyu,
“Learning semantic neural tree for human parsing,” in European
Conference on Computer Vision (ECCV) . Springer, 2020, pp. 205–221.

 

 
 [52] 
 
S. Zhang, G.-J. Qi, X. Cao, Z. Song, and J. Zhou, “Human parsing with
pyramidical gather-excite context,” IEEE Transactions on Circuits and
Systems for Video Technology , vol. 31, no. 3, pp. 1016–1030, 2020.

 

 
 [53] 
 
X. Zhang, Y. Chen, B. Zhu, J. Wang, and M. Tang, “Part-aware context network
for human parsing,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , June 2020.

 

 
 [54] 
 
A. Loesch and R. Audigier, “Describe me if you can! characterized
instance-level human parsing,” in IEEE Conference on Image Processing
(ICIP) . IEEE, 2021, pp. 2528–2532.

 

 
 [55] 
 
Y. Song, X. Chen, J. Li, and Q. Zhao, “Embedding 3d geometric features for
rigid object part segmentation,” in IEEE International Conference on
Computer Vision (ICCV) , 2017, pp. 580–588.

 

 
 [56] 
 
P. Wang, X. Shen, Z. Lin, S. Cohen, B. Price, and A. L. Yuille, “Joint object
and part segmentation using deep learned potentials,” in IEEE
International Conference on Computer Vision (ICCV) , 2015, pp. 1573–1581.

 

 
 [57] 
 
S. Naha, Q. Xiao, P. Banik, M. A. Reza, and D. J. Crandall, “Part segmentation
of unseen objects using keypoint guidance,” in Proceedings of the
IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) , 2021,
pp. 1742–1750.

 

 
 [58] 
 
Z. Wu, G. Lin, and J. Cai, “Keypoint based weakly supervised human parsing,”
 Image and Vision Computing , vol. 91, p. 103801, 2019.

 

 
 [59] 
 
Z. Yang, Y. Li, L. Yang, N. Zhang, and J. Luo, “Weakly supervised body part
segmentation with pose based part priors,” in 2020 25th International
Conference on Pattern Recognition (ICPR) . IEEE, 2021, pp. 286–293.

 

 
 [60] 
 
Y. Zhao, J. Li, Y. Zhang, and Y. Tian, “From pose to part: Weakly-supervised
pose evolution for human part segmentation,” IEEE Transactions on
Pattern Analysis and Machine Intelligence , 2022.

 

 
 [61] 
 
Y. Yang, X. Cheng, H. Bilen, and X. Ji, “Learning to annotate part
segmentation with gradient matching,” in International Conference on
Learning Representations , 2021.

 

 
 [62] 
 
A. Gonzalez-Garcia, D. Modolo, and V. Ferrari, “Do semantic parts emerge in
convolutional neural networks?” International Journal of Computer
Vision , vol. 126, no. 5, pp. 476–494, 2018.

 

 
 [63] 
 
D. Lorenz, L. Bereska, T. Milbich, and B. Ommer, “Unsupervised part-based
disentangling of object shape and appearance,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , 2019, pp. 10 955–10 964.

 

 
 [64] 
 
W.-C. Hung, V. Jampani, S. Liu, P. Molchanov, M.-H. Yang, and J. Kautz,
“Scops: Self-supervised co-part segmentation,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , 2019, pp. 869–878.

 

 
 [65] 
 
Q. Gao, B. Wang, L. Liu, and B. Chen, “Unsupervised co-part segmentation
through assembly,” in International Conference on Machine Learning
(ICML) . PMLR, 2021, pp. 3576–3586.

 

 
 [66] 
 
S. Liu, L. Zhang, X. Yang, H. Su, and J. Zhu, “Unsupervised part segmentation
through disentangling appearance and shape,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , June 2021, pp. 8355–8364.

 

 
 [67] 
 
S. Choudhury, I. Laina, C. Rupprecht, and A. Vedaldi, “Unsupervised part
discovery from contrastive reconstruction,” Advances in Neural
Information Processing Systems (NeurIPS) , vol. 34, 2021.

 

 
 [68] 
 
U. Michieli, E. Borsato, L. Rossi, and P. Zanuttigh, “Gmnet: Graph matching
network for large scale part semantic segmentation in the wild,” in
 European Conference on Computer Vision (ECCV) , 2020, pp. 397–414.

 

 
 [69] 
 
X. Tan, J. Xu, Z. Ye, J. Hao, and L. Ma, “Confident semantic ranking loss for
part parsing,” in 2021 IEEE International Conference on Multimedia and
Expo (ICME) . IEEE, 2021, pp. 1–6.

 

 
 [70] 
 
R. Singh, P. Gupta, P. Shenoy, and R. Sarvadevabhatla, “Float: Factorized
learning of object attributes for improved multi-object multi-part scene
parsing,” IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) , 2022.

 

 
 [71] 
 
A. Kirillov, K. He, R. Girshick, C. Rother, and P. Dollár, “Panoptic
segmentation,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2019, pp. 9404–9413.

 

 
 [72] 
 
M. Everingham, S. M. A. Eslami, L. Van Gool, C. K. I. Williams, J. Winn, and
A. Zisserman, “The pascal visual object classes challenge: A
retrospective,” in IEEE International Conference on Computer Vision
(ICCV) , vol. 111, no. 1, 2015, pp. 98–136.

 

 
 [73] 
 
M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson,
U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset for semantic
urban scene understanding,” in IEEE Conference on Computer Vision and
Pattern Recognition (CVPR) , 2016.

 

 
 [74] 
 
P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ramanan, “Object
detection with discriminatively trained part-based models,” IEEE
Transactions on Pattern Analysis and Machine Intelligence , vol. 32, no. 9,
pp. 1627–1645, 2010.

 

 
 [75] 
 
S. Eslami and C. Williams, “A generative model for parts-based object
segmentation,” in Advances in Neural Information Processing Systems
(NeurIPS) , 2012, pp. 100–107.

 

 
 [76] 
 
S. Liu, J. Feng, C. Domokos, H. Xu, J. Huang, Z. Hu, and S. Yan, “Fashion
parsing with weak color-category labels,” IEEE Transactions on
Multimedia , vol. 16, no. 1, pp. 253–265, 2014.

 

 
 [77] 
 
F. Meng, H. Li, Q. Wu, K. N. Ngan, and J. Cai, “Seeds-based part segmentation
by seeds propagation and region convexity decomposition,” IEEE
Transactions on Multimedia , vol. 20, no. 2, pp. 310–322, 2017.

 

 
 [78] 
 
C. Desai and D. Ramanan, “Detecting actions, poses, and objects with
relational phraselets,” in European Conference on Computer Vision
(ECCV) . Springer, 2012, pp. 158–172.

 

 
 [79] 
 
H. Azizpour and I. Laptev, “Object detection using strongly-supervised
deformable part models,” in European Conference on Computer Vision
(ECCV) , 2012, pp. 836–849.

 

 
 [80] 
 
J. Dong, Q. Chen, Z. Huang, J. Yang, and S. Yan, “Parsing based on parselets:
A unified deformable mixture model for human parsing,” IEEE
Transactions on Pattern Analysis and Machine Intelligence , vol. 38, no. 1,
pp. 88–101, 2015.

 

 
 [81] 
 
F. Xia, J. Zhu, P. Wang, and A. Yuille, “Pose-guided human parsing by an
and/or graph using pose-context features,” in AAAI Conference on
Artificial Intelligence (AAAI) , vol. 30, no. 1, 2016.

 

 
 [82] 
 
K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2016, pp. 770–778.

 

 
 [83] 
 
K. Simonyan and A. Zisserman, “Very deep convolutional networks for
large-scale image recognition,” arXiv preprint arXiv:1409.1556 , 2014.

 

 
 [84] 
 
K. He, X. Zhang, S. Ren, and J. Sun, “Delving deep into rectifiers: Surpassing
human-level performance on imagenet classification,” in IEEE
International Conference on Computer Vision (ICCV) , 2015, pp. 1026–1034.

 

 
 [85] 
 
J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for
semantic segmentation,” in IEEE Conference on Computer Vision and
Pattern Recognition (CVPR) , 2015, pp. 3431–3440.

 

 
 [86] 
 
Y. Yang and D. Ramanan, “Articulated pose estimation with flexible
mixtures-of-parts,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2011, pp. 1385–1392.

 

 
 [87] 
 
J. Dong, Q. Chen, X. Shen, J. Yang, and S. Yan, “Towards unified human parsing
and pose estimation,” in IEEE Conference on Computer Vision and
Pattern Recognition (CVPR) , 2014, pp. 843–850.

 

 
 [88] 
 
L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, “Deeplab:
Semantic image segmentation with deep convolutional nets, atrous convolution,
and fully connected crfs,” IEEE Transactions on Pattern Analysis and
Machine Intelligence , vol. 40, no. 4, pp. 834–848, 2018.

 

 
 [89] 
 
L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam, “Rethinking atrous
convolution for semantic image segmentation,” arXiv:1706.05587 , 2017.

 

 
 [90] 
 
X. Liang, X. Shen, J. Feng, L. Lin, and S. Yan, “Semantic object parsing with
graph lstm,” in European Conference on Computer Vision (ECCV) . Springer, 2016, pp. 125–143.

 

 
 [91] 
 
C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, “The caltech-ucsd
birds-200-2011 dataset,” 2011.

 

 
 [92] 
 
M.-E. Nilsback and A. Zisserman, “Automated flower classification over a large
number of classes,” in 2008 Sixth Indian Conference on Computer
Vision, Graphics Image Processing . IEEE, 2008, pp. 722–729.

 

 
 [93] 
 
A. Khosla, N. Jayadevaprakash, B. Yao, and F.-F. Li, “Novel dataset for
fine-grained image categorization: Stanford dogs,” in Proc. CVPR
workshop on fine-grained visual categorization (FGVC) , vol. 2, no. 1. Citeseer, 2011.

 

 
 [94] 
 
J. Krause, M. Stark, J. Deng, and L. Fei-Fei, “3d object representations for
fine-grained categorization,” in Proceedings of the IEEE International
Conference on Computer Vision Workshops , 2013, pp. 554–561.

 

 
 [95] 
 
S. Maji, E. Rahtu, J. Kannala, M. Blaschko, and A. Vedaldi, “Fine-grained
visual classification of aircraft,” arXiv preprint arXiv:1306.5151 ,
2013.

 

 
 [96] 
 
L. Bossard, M. Guillaumin, and L. V. Gool, “Food-101–mining discriminative
components with random forests,” in European Conference on Computer
Vision (ECCV) . Springer, 2014, pp.
446–461.

 

 
 [97] 
 
T. Berg, J. Liu, S. Woo Lee, M. L. Alexander, D. W. Jacobs, and P. N.
Belhumeur, “Birdsnap: Large-scale fine-grained visual categorization of
birds,” in IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) , 2014, pp. 2011–2018.

 

 
 [98] 
 
G. Van Horn, S. Branson, R. Farrell, S. Haber, J. Barry, P. Ipeirotis,
P. Perona, and S. Belongie, “Building a bird recognition app and large scale
dataset with citizen scientists: The fine print in fine-grained dataset
collection,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2015, pp. 595–604.

 

 
 [99] 
 
L. Yang, P. Luo, C. Change Loy, and X. Tang, “A large-scale car dataset for
fine-grained categorization and verification,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , 2015, pp. 3973–3981.

 

 
 [100] 
 
Z. Liu, P. Luo, S. Qiu, X. Wang, and X. Tang, “Deepfashion: Powering robust
clothes recognition and retrieval with rich annotations,” in IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) , 2016, pp.
1096–1104.

 

 
 [101] 
 
G. Van Horn, O. Mac Aodha, Y. Song, Y. Cui, C. Sun, A. Shepard, H. Adam,
P. Perona, and S. Belongie, “The inaturalist species classification and
detection dataset,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2018, pp. 8769–8778.

 

 
 [102] 
 
M. Sun, Y. Yuan, F. Zhou, and E. Ding, “Multi-attention multi-class constraint
for fine-grained image recognition,” in European Conference on
Computer Vision (ECCV) , 2018, pp. 805–821.

 

 
 [103] 
 
G. Van Horn, E. Cole, S. Beery, K. Wilber, S. Belongie, and O. Mac Aodha,
“Benchmarking representation learning for natural world image collections,”
in IEEE Conference on Computer Vision and Pattern Recognition (CVPR) ,
2021, pp. 12 884–12 893.

 

 
 [104] 
 
P. Zhuang, Y. Wang, and Y. Qiao, “Wildfish: A large benchmark for fish
recognition in the wild,” in ACM International Conference on
Multimedia , 2018, pp. 1301–1309.

 

 
 [105] 
 
T. Weyand, A. Araujo, B. Cao, and J. Sim, “Google landmarks dataset v2-a
large-scale benchmark for instance-level recognition and retrieval,” in
 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) ,
2020, pp. 2575–2584.

 

 
 [106] 
 
B. Yao, A. Khosla, and L. Fei-Fei, “Combining randomization and discrimination
for fine-grained image categorization,” in IEEE Conference on Computer
Vision and Pattern Recognition (CVPR) . IEEE, 2011, pp. 1577–1584.

 

 
 [107] 
 
B. Yao, G. Bradski, and L. Fei-Fei, “A codebook-free and annotation-free
approach for fine-grained image categorization,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) . IEEE, 2012, pp. 3466–3473.

 

 
 [108] 
 
C. Goring, E. Rodner, A. Freytag, and J. Denzler, “Nonparametric part transfer
for fine-grained recognition,” in IEEE Conference on Computer Vision
and Pattern Recognition (CVPR) , 2014, pp. 2489–2496.

 

 
 [109] 
 
C. Wah, S. Branson, P. Perona, and S. Belongie, “Multiclass recognition and
part localization with humans in the loop,” in IEEE International
Conference on Computer Vision (ICCV) . IEEE, 2011, pp. 2524–2531.

 

 
 [110] 
 
B. He, J. Li, Y. Zhao, and Y. Tian, “Part-regularized near-duplicate vehicle
re-identification,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2019, pp. 3997–4005.

 

 
 [111] 
 
Y. Peng, X. He, and J. Zhao, “Object-part attention model for fine-grained
image classification,” IEEE Transactions on Image Processing ,
vol. 27, no. 3, pp. 1487–1500, 2017.

 

 
 [112] 
 
D. Wang, Z. Shen, J. Shao, W. Zhang, X. Xue, and Z. Zhang, “Multiple
granularity descriptors for fine-grained categorization,” in
 Proceedings of the IEEE international conference on computer vision ,
2015, pp. 2399–2406.

 

 
 [113] 
 
J. Krause, H. Jin, J. Yang, and L. Fei-Fei, “Fine-grained recognition without
part annotations,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2015, pp. 5546–5555.

 

 
 [114] 
 
Y. Zhao, K. Yan, F. Huang, and J. Li, “Graph-based high-order relation
discovery for fine-grained recognition,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2021, pp.
15 079–15 088.

 

 
 [115] 
 
T.-Y. Lin, A. RoyChowdhury, and S. Maji, “Bilinear cnn models for fine-grained
visual recognition,” in IEEE International Conference on Computer
Vision (ICCV) , 2015, pp. 1449–1457.

 

 
 [116] 
 
Y. Gao, O. Beijbom, N. Zhang, and T. Darrell, “Compact bilinear pooling,” in
 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) ,
2016, pp. 317–326.

 

 
 [117] 
 
Y. Li, N. Wang, J. Liu, and X. Hou, “Factorized bilinear models for image
recognition,” in IEEE International Conference on Computer Vision
(ICCV) , 2017, pp. 2079–2087.

 

 
 [118] 
 
X. Wei, Y. Zhang, Y. Gong, J. Zhang, and N. Zheng, “Grassmann pooling as
compact homogeneous bilinear pooling for fine-grained visual
classification,” in European Conference on Computer Vision (ECCV) ,
2018, pp. 355–370.

 

 
 [119] 
 
H. Zheng, J. Fu, Z.-J. Zha, and J. Luo, “Looking for the devil in the details:
Learning trilinear attention sampling network for fine-grained image
recognition,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2019, pp. 5012–5021.

 

 
 [120] 
 
Y. Gao, X. Han, X. Wang, W. Huang, and M. Scott, “Channel interaction networks
for fine-grained image categorization,” in AAAI Conference on
Artificial Intelligence (AAAI) , 2020, pp. 10 818–10 825.

 

 
 [121] 
 
X. Wang, R. Girshick, A. Gupta, and K. He, “Non-local neural networks,” in
 Proceedings of the IEEE conference on computer vision and pattern
recognition , 2018, pp. 7794–7803.

 

 
 [122] 
 
M. Simon and E. Rodner, “Neural activation constellations: Unsupervised part
model discovery with convolutional networks,” in IEEE International
Conference on Computer Vision (ICCV) , 2015, pp. 1143–1151.

 

 
 [123] 
 
Y. Zhang, X.-S. Wei, J. Wu, J. Cai, J. Lu, V.-A. Nguyen, and M. N. Do, “Weakly
supervised fine-grained categorization with part-based image
representation,” IEEE Transactions on Image Processing , vol. 25,
no. 4, pp. 1713–1725, 2016.

 

 
 [124] 
 
J. Fu, H. Zheng, and T. Mei, “Look closer to see better: Recurrent attention
convolutional neural network for fine-grained image recognition,” in
 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) ,
2017, pp. 4438–4446.

 

 
 [125] 
 
A. Recasens, P. Kellnhofer, S. Stent, W. Matusik, and A. Torralba, “Learning
to zoom: a saliency-based sampling layer for neural networks,” in
 European Conference on Computer Vision (ECCV) , 2018, pp. 51–66.

 

 
 [126] 
 
Z. Wang, S. Wang, S. Yang, H. Li, J. Li, and Z. Li, “Weakly supervised
fine-grained image classification via guassian mixture model oriented
discriminative learning,” in IEEE Conference on Computer Vision and
Pattern Recognition (CVPR) , 2020, pp. 9749–9758.

 

 
 [127] 
 
W. Ge, X. Lin, and Y. Yu, “Weakly supervised complementary parts models for
fine-grained image classification from the bottom up,” in IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) , 2019, pp.
3034–3043.

 

 
 [128] 
 
G. Sun, H. Cholakkal, S. Khan, F. Khan, and L. Shao, “Fine-grained
recognition: Accounting for subtle differences between similar classes,” in
 AAAI Conference on Artificial Intelligence (AAAI) , vol. 34, no. 07,
2020, pp. 12 047–12 054.

 

 
 [129] 
 
H. Zheng, J. Fu, Z.-J. Zha, J. Luo, and T. Mei, “Learning rich part
hierarchies with progressive attention networks for fine-grained image
recognition,” IEEE Transactions on Image Processing , vol. 29, pp.
476–488, 2019.

 

 
 [130] 
 
Y. Ding, Y. Zhou, Y. Zhu, Q. Ye, and J. Jiao, “Selective sparse sampling for
fine-grained image recognition,” in IEEE International Conference on
Computer Vision (ICCV) , 2019, pp. 6599–6608.

 

 
 [131] 
 
Z. Wang, S. Wang, H. Li, Z. Dou, and J. Li, “Graph-propagation based
correlation learning for weakly supervised fine-grained image
classification,” in AAAI Conference on Artificial Intelligence
(AAAI) , vol. 34, no. 07, 2020, pp. 12 289–12 296.

 

 
 [132] 
 
M. Lam, B. Mahasseni, and S. Todorovic, “Fine-grained recognition as hsnet
search for informative image parts,” in IEEE Conference on Computer
Vision and Pattern Recognition (CVPR) , 2017, pp. 2520–2529.

 

 
 [133] 
 
Y. Zhao, J. Li, X. Chen, and Y. Tian, “Part-guided relational transformers for
fine-grained visual recognition,” IEEE Transactions on Image
Processing , vol. 30, pp. 9470–9481, 2021.

 

 
 [134] 
 
R. Ji, L. Wen, L. Zhang, D. Du, Y. Wu, C. Zhao, X. Liu, and F. Huang,
“Attention convolutional binary neural tree for fine-grained visual
categorization,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2020, pp. 10 468–10 477.

 

 
 [135] 
 
M. Nauta, R. van Bree, and C. Seifert, “Neural prototype trees for
interpretable fine-grained image recognition,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2021, pp.
14 933–14 943.

 

 
 [136] 
 
Z. Yang, T. Luo, D. Wang, Z. Hu, J. Gao, and L. Wang, “Learning to navigate
for fine-grained classification,” in European Conference on Computer
Vision (ECCV) , 2018, pp. 420–435.

 

 
 [137] 
 
Y. Wang, V. I. Morariu, and L. S. Davis, “Learning a discriminative filter
bank within a cnn for fine-grained recognition,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , 2018, pp. 4148–4157.

 

 
 [138] 
 
A. Dubey, O. Gupta, P. Guo, R. Raskar, R. Farrell, and N. Naik, “Pairwise
confusion for fine-grained visual classification,” in European
Conference on Computer Vision (ECCV) , 2018, pp. 70–86.

 

 
 [139] 
 
O. M. Aodha, E. Cole, and P. Perona, “Presence-only geographical priors for
fine-grained image classification,” in IEEE International Conference
on Computer Vision (ICCV) , 2019, pp. 9595–9605.

 

 
 [140] 
 
A. Dubey, O. Gupta, R. Raskar, and N. Naik, “Maximum-entropy fine grained
classification,” in Advances in Neural Information Processing Systems
(NeurIPS) , 2018, pp. 637–647.

 

 
 [141] 
 
Y. Cui, Y. Song, C. Sun, A. Howard, and S. Belongie, “Large scale fine-grained
categorization and domain-specific transfer learning,” in IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) , 2018, pp.
4109–4118.

 

 
 [142] 
 
X. Zheng, R. Ji, X. Sun, B. Zhang, Y. Wu, and F. Huang, “Towards optimal fine
grained retrieval via decorrelated centralized loss with normalize-scale
layer,” in AAAI Conference on Artificial Intelligence (AAAI) ,
vol. 33, no. 01, 2019, pp. 9291–9298.

 

 
 [143] 
 
X.-S. Wei, J.-H. Luo, J. Wu, and Z.-H. Zhou, “Selective convolutional
descriptor aggregation for fine-grained image retrieval,” IEEE
Transactions on Image Processing , vol. 26, no. 6, pp. 2868–2881, 2017.

 

 
 [144] 
 
C. Yu, X. Zhao, Q. Zheng, P. Zhang, and X. You, “Hierarchical bilinear pooling
for fine-grained visual recognition,” in European Conference on
Computer Vision (ECCV) , 2018, pp. 574–589.

 

 
 [145] 
 
L. Zhang, S. Huang, W. Liu, and D. Tao, “Learning a mixture of
granularity-specific experts for fine-grained categorization,” in IEEE
International Conference on Computer Vision (ICCV) , 2019, pp. 8331–8340.

 

 
 [146] 
 
S. Kong and C. Fowlkes, “Low-rank bilinear pooling for fine-grained
classification,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2017, pp. 365–374.

 

 
 [147] 
 
Y. Chen, Y. Bai, W. Zhang, and T. Mei, “Destruction and construction learning
for fine-grained image recognition,” in IEEE Conference on Computer
Vision and Pattern Recognition (CVPR) , 2019, pp. 5157–5166.

 

 
 [148] 
 
W. Luo, X. Yang, X. Mo, Y. Lu, L. S. Davis, J. Li, J. Yang, and S.-N. Lim,
“Cross-x learning for fine-grained visual categorization,” in IEEE
International Conference on Computer Vision (ICCV) , 2019, pp. 8242–8251.

 

 
 [149] 
 
P. Zhuang, Y. Wang, and Y. Qiao, “Learning attentive pairwise interaction for
fine-grained classification,” in AAAI Conference on Artificial
Intelligence (AAAI) , vol. 34, no. 07, 2020, pp. 13 130–13 137.

 

 
 [150] 
 
C. Liu, H. Xie, Z. Zha, L. Yu, Z. Chen, and Y. Zhang, “Bidirectional
attention-recognition model for fine-grained object classification,”
 IEEE Transactions on Multimedia , vol. 22, no. 7, pp. 1785–1795, 2019.

 

 
 [151] 
 
P. Rodríguez, J. M. Gonfaus, G. Cucurull, F. XavierRoca, and J. Gonzalez,
“Attend and rectify: a gated attention mechanism for fine-grained
recovery,” in European Conference on Computer Vision (ECCV) , 2018,
pp. 349–364.

 

 
 [152] 
 
C. Liu, H. Xie, Z.-J. Zha, L. Ma, L. Yu, and Y. Zhang, “Filtration and
distillation: Enhancing region attention for fine-grained visual
categorization,” in AAAI Conference on Artificial Intelligence
(AAAI) , vol. 34, no. 07, 2020, pp. 11 555–11 562.

 

 
 [153] 
 
D. Wu, S. Li, Z. Zang, K. Wang, L. Shang, B. Sun, H. Li, and S. Z. Li, “Align
yourself: Self-supervised pre-training for fine-grained recognition via
saliency alignment,” arXiv preprint arXiv:2106.15788 , 2021.

 

 
 [154] 
 
J. Wang, Y. Li, X.-S. Wei, H. Li, Z. Miao, and R. Zhang, “Bridge the gap
between supervised and unsupervised learning for fine-grained
classification,” arXiv preprint arXiv:2203.00441 , 2022.

 

 
 [155] 
 
D. Chang, K. Pang, Y. Zheng, Z. Ma, Y.-Z. Song, and J. Guo, “Your” flamingo”
is my” bird”: Fine-grained, or not,” in IEEE Conference on Computer
Vision and Pattern Recognition (CVPR) , 2021, pp. 11 476–11 485.

 

 
 [156] 
 
M. Zhou, Y. Bai, W. Zhang, T. Zhao, and T. Mei, “Look-into-object:
Self-supervised structure modeling for object recognition,” in IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) , 2020, pp.
11 774–11 783.

 

 
 [157] 
 
M. M. Kalayeh, E. Basaran, M. Gökmen, M. E. Kamasak, and M. Shah, “Human
semantic parsing for person re-identification,” in Proceedings of the
IEEE conference on computer vision and pattern recognition , 2018, pp.
1062–1071.

 

 
 [158] 
 
D. Meng, L. Li, X. Liu, Y. Li, S. Yang, Z.-J. Zha, X. Gao, S. Wang, and
Q. Huang, “Parsing-based view-aware embedding network for vehicle
re-identification,” in Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition , 2020, pp. 7103–7112.

 

 
 [159] 
 
J. Zhao, Y. Zhao, J. Li, K. Yan, and Y. Tian, “Heterogeneous relational
complement for vehicle re-identification,” in Proceedings of the
IEEE/CVF International Conference on Computer Vision , 2021, pp. 205–214.

 

 
 [160] 
 
W.-C. Chen, X.-Y. Yu, and L.-L. Ou, “Pedestrian attribute recognition in video
surveillance scenarios based on view-attribute attention localization,”
 Machine Intelligence Research , vol. 19, no. 2, pp. 153–168, 2022.

 

 
 [161] 
 
B. M. Lake, R. Salakhutdinov, and J. B. Tenenbaum, “Human-level concept
learning through probabilistic program induction,” Science , vol. 350,
no. 6266, pp. 1332–1338, 2015.

 

 
 [162] 
 
L. Fei-Fei, R. Fergus, and P. Perona, “One-shot learning of object
categories,” IEEE Transactions on Pattern Analysis and Machine
Intelligence , vol. 28, no. 4, pp. 594–611, 2006.

 

 
 [163] 
 
E. G. Miller, N. E. Matsakis, and P. A. Viola, “Learning from one example
through shared densities on transforms,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , vol. 1. IEEE, 2000, pp. 464–471.

 

 
 [164] 
 
L. Tang, D. Wertheimer, and B. Hariharan, “Revisiting pose-normalization for
fine-grained few-shot recognition,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , 2020, pp.
14 352–14 361.

 

 
 [165] 
 
P. Koniusz and H. Zhang, “Power normalizations in fine-grained image, few-shot
image and graph classification,” IEEE Transactions on Pattern Analysis
and Machine Intelligence , vol. 44, no. 2, pp. 591–609, 2021.

 

 
 [166] 
 
H. Huang, J. Zhang, J. Zhang, J. Xu, and Q. Wu, “Low-rank pairwise alignment
bilinear network for few-shot fine-grained image classification,” IEEE
Transactions on Multimedia , vol. 23, pp. 1666–1680, 2020.

 

 
 [167] 
 
Y. Zhu, C. Liu, and S. Jiang, “Multi-attention meta learning for few-shot
fine-grained image recognition.” in IJCAI , 2020, pp. 1090–1096.

 

 
 [168] 
 
A.-X. Li, K.-X. Zhang, and L.-W. Wang, “Zero-shot fine-grained classification
by deep feature learning with semantics,” International Journal of
Automation and Computing , vol. 16, no. 5, pp. 563–574, 2019.

 

 
 [169] 
 
X. Zhang, F. Zhou, Y. Lin, and S. Zhang, “Embedding label structures for
fine-grained feature representation,” in Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition , 2016, pp. 1114–1123.

 

 
 [170] 
 
S. Xie, T. Yang, X. Wang, and Y. Lin, “Hyper-class augmented and regularized
deep learning for fine-grained image classification,” in IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) , 2015, pp.
2645–2654.

 

 
 [171] 
 
A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry,
A. Askell, P. Mishkin, J. Clark et al. , “Learning transferable visual
models from natural language supervision,” in International Conference
on Machine Learning (ICML) . PMLR,
2021, pp. 8748–8763.

 

 
 [172] 
 
L. H. Li, P. Zhang, H. Zhang, J. Yang, C. Li, Y. Zhong, L. Wang, L. Yuan,
L. Zhang, J.-N. Hwang et al. , “Grounded language-image
pre-training,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2022, pp. 10 965–10 975.

 

 
 [173] 
 
E. Kalogerakis, M. Averkiou, S. Maji, and S. Chaudhuri, “3d shape segmentation
with projective convolutional networks,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , 2017, pp. 3779–3788.

 

 
 [174] 
 
F. Yu, K. Liu, Y. Zhang, C. Zhu, and K. Xu, “Partnet: A recursive part
decomposition network for fine-grained and hierarchical shape segmentation,”
in IEEE Conference on Computer Vision and Pattern Recognition (CVPR) ,
2019, pp. 9491–9500.

 

 
 [175] 
 
K. Mo, S. Zhu, A. X. Chang, L. Yi, S. Tripathi, L. J. Guibas, and H. Su,
“Partnet: A large-scale benchmark for fine-grained and hierarchical
part-level 3d object understanding,” in IEEE Conference on Computer
Vision and Pattern Recognition (CVPR) , 2019, pp. 909–918.

 

 
 [176] 
 
C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point
sets for 3d classification and segmentation,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , 2017, pp. 652–660.

 

 
 [177] 
 
J. Kittler, A. Hilton, M. Hamouz, and J. Illingworth, “3d assisted face
recognition: A survey of 3d imaging, modelling and recognition approachest,”
in 2005 IEEE Computer Society Conference on Computer Vision and Pattern
Recognition (CVPR’05)-Workshops . IEEE, 2005, pp. 114–114.

 

 
 [178] 
 
S. Joung, S. Kim, M. Kim, I.-J. Kim, and K. Sohn, “Learning canonical 3d
object representation for fine-grained recognition,” in IEEE
International Conference on Computer Vision (ICCV) , 2021, pp. 1035–1045.

 

 
 [179] 
 
B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and
R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM , vol. 65, no. 1, pp. 99–106,
2021.

 

 
 
 
 
 
 
 | 
 
 
 Yifan Zhao is currently a postdoctoral researcher with the School of Computer Science, Peking University, Beijing, China. He received the B.E. degree from Harbin Institute of Technology in Jul. 2016 and the Ph.D. degree from the School of Computer Science and Engineering, Beihang University, in Nov. 2021. His research interests include computer vision and image/video understanding. 
 | 

 
 
 
 
 | 
 
 
 Jia Li (M’12-SM’15) is currently a Full Professor with the State Key Laboratory of Virtual Reality Technology and Systems, School of Computer Science and Engineering, Beihang University. He received his B.E. degree from Tsinghua University in 2005 and Ph.D. degree from Institute of Computing Technology, Chinese Academy of Sciences, in 2011. Before he joined Beihang University in 2014, he used to work at Nanyang Technological University, Shanda Innovations, and Peking University. His research is focused on computer vision, multimedia and artificial intelligence, especially the visual computing in extreme environments. He has co-authored more than 110 articles in peer-reviewed top-tier journals and conferences. He also has one Monograph published by Springer and more than 60 patents issued from U.S. and China. He is a Fellow of IET, and senior members of IEEE/ACM/CCF/CIE. 
 | 

 
 
 
 
 | 
 
 
 Yonghong Tian (S’00-M’06-SM’10) is currently a Boya Distinguished Professor with the School of Computer Science, Peking University, China, and is also the deputy director of Artificial Intelligence Research Center, PengCheng Laboratory, Shenzhen, China. His research interests include neuromorphic vision, distributed machine learning and multimedia big data. He is the author or coauthor of over 280 technical articles in refereed journals and conferences. Prof. Tian was/is an Associate Editor of IEEE TCSVT (2018.1-2021.12), IEEE TMM (2014.8-2018.8), IEEE Multimedia Mag. (2018.1-), and IEEE Access (2017.1-). He co-initiated IEEE Intl Conf. on Multimedia Big Data (BigMM) and served as the TPC Co-chair of BigMM 2015, and aslo served as the Technical Program Co-chair of IEEE ICME 2015, IEEE ISM 2015 and IEEE MIPR 2018/2019, and General Co-chair of IEEE MIPR 2020 and ICME 2021. He is the steering member of IEEE ICME (2018-2020) and IEEE BigMM (2015-), and is a TPC Member of more than ten conferences such as CVPR, ICCV, ACM KDD, AAAI, ACM MM and ECCV. He was the recipient of the Chinese National Science Foundation for Distinguished Young Scholars in 2018, two National Science and Technology Awards and three ministerial-level awards in China, and obtained the 2015 EURASIP Best Paper Award for Journal on Image and Video Processing, and the best paper award of IEEE BigMM 2018. He is a Fellow of IEEE, a senior member of CIE and CCF, a member of ACM. 
 |